# Godhaar Inference Server — working notes

FastAPI + PyTorch (CUDA) service that detects, embeds, and matches cattle
muzzles. It is called only by the API server (never the app directly), returns
raw similarity scores, and knows nothing about farmers/ownership. Endpoints:
`POST /register` (201), `POST /search`, `GET /health`. Port 9050.

There is now a **third model**, `cctv/` — video-based cattle counting/
tracking/analytics, mounted under `/cctv` in `main.py`. It is fully
self-contained (own config, SQLite session DB, in-memory background job
queue) and independent of the register/search decision chain — no GPS,
no farmer context, no FAISS. See the README's "Video Analytics (CCTV
Model)" section for the endpoint list. Ported from a standalone prototype
(`D:\Group Projects\cattle_ai_project`) that had already been built and
verified against real sample video before the port.

### Tracker YAML compatibility: `cmc_method` → `gmc_method` (ultralytics renamed the key)

Porting the CCTV model's BoT-SORT tracker configs
(`cctv/trackers/botsort_cattle_fast.yaml` / `botsort_cattle.yaml`) verbatim
from the source prototype caused every `/cctv/analyze` job to fail
mid-processing with `'IterableSimpleNamespace' object has no attribute
'gmc_method'`. The prototype was built against `ultralytics>=8.3` (loose);
this repo pins `ultralytics==8.4.83`. Between those versions ultralytics
renamed BoT-SORT's camera-motion-compensation key from `cmc_method` to
`gmc_method` (confirmed by reading the installed package's own
`ultralytics/cfg/trackers/botsort.yaml`) — the old key is now silently
ignored by YAML parsing (not a KeyError) and the code that reads
`args.gmc_method` finds nothing, so the failure only surfaces once BoT-SORT
actually runs, not at config-load time. Fixed by renaming the key in both
YAMLs and adding `model: auto` (the installed default's value, needed once
`with_reid: true` is set for the accurate preset). **Verified end-to-end
after the fix**: ran a real sample video through `/cctv/analyze` on this
machine's GPU — 16 unique cattle tracked, full analytics (density grid,
per-cow speed/activity, isolation) computed, session persisted to SQLite,
annotated video downloadable via `/cctv/jobs/{id}/video`.

**Lesson: never trust a bundled tracker/model YAML to survive a version
bump untouched — diff it against the currently-installed library's own
default config before assuming it will load.** This is the same class of
mistake as pinning a numeric threshold across two codebases without
checking they mean the same thing (see the Telangana app's CLAUDE.md for
other examples of this pattern) — here it was a config *key name*, not a
value, but the failure mode (silently wrong until the code path actually
runs) is identical.

Also fixed while porting: the prototype's `cattle_ai/analytics.py` called
`cv2.applyColorMap`/`cv2.line` in `_render_heatmap`/`draw_trajectories`
without ever importing `cv2` — a `NameError` waiting to happen on the very
first `compute()` call with `enable_analytics=True` (which is the default).
Never triggered in the prototype because whatever testing it got apparently
didn't exercise that path end-to-end. Added the import in `cctv/analytics.py`.

## Cross-camera de-duplication: muzzle-crop extraction built; comparison logic deliberately NOT built yet

Every CCTV job counts cattle **per video**, independently. Two cameras
covering the same or adjacent physical space — e.g. an entry gate and the
shed it feeds into — have no way to know they might be looking at the same
animal a few seconds apart, so a goshala running several cameras has no
real "how many distinct cattle did we actually see today" number, only a
pile of per-camera counts that may double-count animals. This section
covers what's built (a layout-independent prerequisite, safe to ship before
any camera's physical position is known) and what's designed but
deliberately not built (the actual cross-camera match, which DOES need
that layout).

### Part 1 — built: one muzzle-crop extraction per tracked cow (`cctv/muzzle_crop.py`)

Before two sightings can ever be compared, each one needs a crop to compare
*with*. `StableIdMapper` (`cctv/stable_id.py`) had zero connection to the
muzzle-biometric side of this server before this — no crop, no `faiss_id`,
nothing. `cctv/muzzle_crop.py` adds exactly one thing: for every stable ID
that survives the existing `min_frames_visible` flicker filter (the same
filter `unique_tracked_cattle` already uses), extract one crop from its
single highest-tracker-confidence sighting across the whole clip.

**There is no muzzle-only detector to reuse — it doesn't exist.**
`pipeline/muzzle_detect.py` is a permanent no-op (CTO-confirmed 2026-08-17,
see `godhaar/config.py`'s "Muzzle detector: removed, permanently absent by
design"). The only thing that has EVER localized a "muzzle crop" for
either `/register` or `/search`, for the actual embedding path, is
`crop_cattle()` (`pipeline/yolo_crop.py`) — a whole-animal COCO detector
that reads as muzzle-filling only because the capture UI demands a 20-30cm
close-up (see "The pipeline embeds the whole animal" above). So reusing
"the same crop_cattle/muzzle-detection logic already used for
registration" means, concretely, reusing `crop_cattle()` and nothing else
— that IS the whole of what registration has.

A CCTV frame is not a close-up, and a goshala frame usually has several
cattle in it. Calling `crop_cattle()` on the full frame would re-create the
exact multi-subject ambiguity `select_dominant_box()` exists to resolve for
registration photos — except CCTV doesn't need that heuristic at all,
because BoT-SORT tracking already says exactly which pixels are OUR cow.
So `cctv/muzzle_crop.py` pre-crops the frame to the tracked box (padded by
`CONTEXT_PADDING_FRACTION = 0.25` of the box's own width/height, minimum 20
px, so `crop_cattle()`'s own YOLO pass has slack to re-detect the animal
rather than clip a leg or horn off the edge it's handed) and only then
calls `crop_cattle()` on that sub-image, unmodified. The result is the same
KIND of crop registration produces — a whole-animal-dominated JPEG, called
"muzzle crop" by this codebase's convention — not a true muzzle-only
region, because that region no longer exists as a concept anywhere in this
server. **`CONTEXT_PADDING_FRACTION` is an unvalidated starting guess**,
same status as the Telangana app's `CONTEXT_MULTIPLIER=4` for the identical
class of problem (see that repo's CLAUDE.md) — if extraction keeps clipping
limbs/horns, raise it; if it keeps pulling in a neighbouring animal, lower
it.

Storage: one JPEG per qualifying stable ID, at
`cctv/runs/<job_id>/muzzle_crops/<stable_id>.jpg` — keyed purely by
`(job_id, stable_id)`, i.e. the CCTV session and the tracked cow, **never**
by `faiss_id` or any registered-animal identity. This has to work for
cattle that were never registered at all: cross-camera de-dup is "same
physical animal across two feeds," not "same registered identity" (see
Part 2). `VideoSummary.muzzle_crops` (new field, `cctv/pipeline.py`) carries
the per-cow result — `status` (`crop_cattle()`'s own status string, or
`NO_SIGHTING`/`DEGENERATE_BOX`/`WRITE_FAILED`/`ERROR`), `crop_path`,
`source_confidence` (the TRACKER's confidence, not `crop_cattle()`'s),
`crop_confidence`, `source_frame_idx` — through to `report.json` the same
way every other summary field already does. `PipelineConfig.
extract_muzzle_crops` (default `True`) gates it; `run_classify_and_count()`
turns it off for its throwaway FAST-preset classification pass, whose
output directory is deleted immediately after anyway (see "Classification
and counting run at different resolutions" below).

**Deliberately NOT wired into `cctv/database.py`'s `tracks` table, or any
route/schema.** `tracks` is already exactly `(job_id, stable_id)` and would
be the obvious place to persist a crop path long-term, but there is no
consumer for that yet — Part 2 below is design only, not built — and
wiring persistence for a value nothing reads yet is the kind of premature
plumbing this file elsewhere warns against. When Part 2 actually gets
built, adding a `muzzle_crop_path`/`muzzle_crop_status` column to `tracks`
(populated in `db.save_session()` from `summary.muzzle_crops`, mirroring
exactly how `analytics.per_cow` already populates that table) is the
natural next step — not a redesign.

**Tested against the real 23-clip goshala set** (`cctv/clips/`, same
government CCTV footage the `CROWDED_HD` preset was calibrated against —
see "Classification and counting run at different resolutions" below) via
`scratch_cctv_muzzle_crop_test.py`, FAST preset:

- **812 qualifying tracked cattle across 23 clips, 633 crops successfully
  extracted — 78.0% overall.** Every clip produced at least one crop; no
  clip's pipeline run failed outright. Per-clip rate ranged 58%-97%, no
  correlation with clip length or `unique_tracked_cattle` count visible at
  a glance (a 24-cow clip scored 58%, a 29-cow clip scored 97%).
- **Failure breakdown: 126 `RECAPTURE_NO_DETECTION` (15.5%), 53
  `RECAPTURE_MULTI_CATTLE` (6.5%), zero of any other status.** No
  `DEGENERATE_BOX`/`WRITE_FAILED`/`ERROR` at all — the padding/geometry
  code path itself never broke; every failure was `crop_cattle()` itself
  declining, the same two rejection reasons it already gives real
  registration photos.
  - `RECAPTURE_NO_DETECTION` skews toward the tracker's own
    lower-confidence sightings (spot-checked several — most sat in the
    0.4-0.6 tracker-confidence band, well below the 0.8+ where most
    `OK` results cluster). Reads as the honest case: a cow's single BEST
    frame in a 30-second clip can still be small, angled away, or
    partially occluded, and `crop_cattle()` correctly declines rather than
    guessing.
  - `RECAPTURE_MULTI_CATTLE` is exactly the goshala-crowding case
    `select_dominant_box()` exists for — `CONTEXT_PADDING_FRACTION=0.25`'s
    padding sometimes pulls a neighbour into the sub-image clearly enough
    that no single animal dominates it.
- Spot-checked output files directly (not just counting successes): crop
  count on disk matched the reported success count exactly (633 files), all
  real, non-trivial JPEGs (6-37 KB sampled, not empty/corrupt writes).
- ~118-121s/clip end-to-end on this machine's GPU (FAST preset, ~30-45s
  clips) — extraction adds a second YOLO pass (`crop_cattle`'s own model,
  separate from the CCTV tracker's) per qualifying cow on top of the
  existing tracking pass; not benchmarked in isolation against a
  crop-extraction-disabled run, so the exact marginal cost isn't broken out
  here.

**78% is a first real measurement, not a target that's been tuned toward.**
The two failure modes are both `crop_cattle()` correctly declining a
genuinely hard case, not a bug in the new code — so the honest next levers,
if this rate ever needs to be higher, are the ones already named inline
above (`CONTEXT_PADDING_FRACTION` for the multi-cattle case) or accepting
that ~1 in 5 tracked cattle in a 30-45s clip simply never gets a
good-enough frame, and letting Part 2's eventual matching logic treat a
missing crop as "nothing to compare," not an error.

### Part 2 — design for the actual cross-camera match (NOT built)

This is the plan for the piece Part 1 is prerequisite to, written down now
so it doesn't need re-deriving once Raipur's (or any goshala's) real camera
layout is known. Nothing below is implemented.

**Core mechanism: reuse `pipeline/lightglue_verify.py`'s `verify()` exactly
as it runs for `/search` today, pointed at two CCTV crops instead of a
query-vs-FAISS-candidate pair.** `/search`'s tiebreaker calls
`verify_with_cached_candidate(query_crop, cached_candidate_features)` — the
"cached" half only exists because the candidate is a *registered* animal
with a `faiss_id`-keyed cache entry (`pipeline/muzzle_crop_cache.py`).
Neither side of a CCTV-to-CCTV comparison has that, so this would call the
plain `verify(crop_a, crop_b)` path instead — full DISK extraction on BOTH
sides, same function `/search`'s own cold-cache fallback and the offline
`experiments/lightglue_poc/match_muzzles.py` already use. `classify_zone()`
and its thresholds (`LIKELY_DIFFERENT_MAX = 140`, `LIKELY_SAME_MIN = 150`)
are the same code too, but **their calibration is NOT known to transfer**:
both numbers were derived from 51 real close-up-registration-photo pairs
(`lightglue_fp_results_v2_resize_cache.csv`, see "/search fusion
tiebreaker" below) — same visual domain as `/register`/`/search`, not the
whole-animal-from-a-distance domain Part 1's crops actually live in. Treat
140/150 as a starting point to re-validate against real CCTV crop pairs,
not an assumption to build on. `LIGHTGLUE_MAX_DIM=512`'s resize cap applies
identically either way, so per-pair latency should still land near the
~170ms measured for `/search` — but every CCTV-to-CCTV pair pays full
two-sided DISK extraction (no feature cache exists for either side), so
that ~170ms is the right per-pair budget to plan N-pair costs against, not
`/search`'s cheaper cached-candidate path.

**⚠️ REQUIRED before Part 3 (the real comparison logic) starts — not
optional, not a nice-to-have: re-calibrate `LIKELY_DIFFERENT_MAX`/
`LIKELY_SAME_MIN` against real CCTV-style crop pairs (whole-animal + 25%
context padding, Part 1's actual output) before writing a single line of
cross-camera matching code against the 140/150 values as they stand.**
Same reasoning as the `FAST`/`CROWDED_HD` confound elsewhere in this file
(a preset's thresholds, tuned for one job, silently corrupted a different
signal the moment it got reused for a second job at a different
resolution/population) — a threshold is only ever validated for the
population it was measured against, and 140/150 were measured against
close-up registration photos, a visual domain Part 1's crops structurally
are not. Do not assume the gap holds; measure it on real CCTV pairs the
same way `/search`'s round-2 re-validation did (real dataset, both
zone-boundary populations checked, not just the happy path).

This isn't hypothetical caution — it's already been shown, in this exact
codebase, on the SAME domain the 140/150 numbers came from, that this kind
of boundary is thinner than a first pass suggests: the "Precision:
LightGlue's actual effect" analysis below (161 real leave-one-out animals,
same registration-photo population 140/150 were calibrated on) found the
138 same-animal pairs currently below 140 include two genuine impostor-free
outliers at 119/136 against an impostor ceiling of exactly **117** —
a real, still-unshipped, config-only improvement (candidate ~118, `140` is
still what's live in `lightglue_verify.py` today) sitting on a measured
2-count margin, WITHIN the domain the threshold was built for. Carrying
140/150 across to a structurally different domain (CCTV whole-body crops)
with zero re-measurement risks a much larger, silent version of the same
gap — possibly in either direction (too many false `likely_different`
demotions, or too many false `likely_same` merges, both of which directly
corrupt a de-duplication count).

**Trigger conditions — explicitly UNDETERMINED, pending real camera data:**
- Same time window across two camera feeds (a sighting on camera A and a
  sighting on camera B are only worth comparing if they could plausibly be
  the same physical moment, not the same cow revisiting hours apart).
- Only for cameras that are physically adjacent or have overlapping
  coverage — comparing every camera's every tracked cow against every
  other camera's is both expensive (N×M DISK+LightGlue pairs, no caching
  possible on either side per the paragraph above) and mostly pointless
  (two cameras on opposite ends of a goshala can't be seeing the same
  animal at the same moment).

**The merge step happens AFTER each camera's own count, as a separate,
goshala-wide de-duplication pass — not a change to per-camera counting.**
`cctv/pipeline.py`'s `process_video()` and `unique_tracked_cattle` stay
exactly what they are today, one number per video. Cross-camera de-dup
would run afterward, across two or more already-completed sessions'
`muzzle_crops` outputs, and produce a SEPARATE goshala-wide figure — it
should never reach back into how one camera's own tracking/counting works.

**Explicitly NOT YET DECIDED — both need real data, not a guess now:**
- **How camera adjacency gets configured.** Likely a manual admin input per
  goshala — something in the shape of `cctv/config.py`'s
  `LOCATION_PRESET_OVERRIDES` (an explicit map keyed by the caller's
  `location_tag`, empty by default, populated only once someone has a real
  reason to populate a specific entry) rather than anything inferred
  automatically. Not designed further than that shape here.
- **What time-window tolerance counts as "plausibly the same sighting."**
  Depends entirely on real walking speed between two specific cameras'
  fields of view at a real goshala, which doesn't exist as data yet.

**This does NOT require the animal to be registered in the muzzle
biometric system at all.** It's tracking-level identity matching — "is the
cow in this crop the same physical animal as the cow in that crop" —
completely independent of whether either animal has ever been through
`/register`/`/search`. That independence is why Part 1 keys crops by
`(job_id, stable_id)` and never by `faiss_id`: a `faiss_id` requires
registration, and most cattle a CCTV camera sees on any given day will not
be registered.

## The pipeline embeds the whole animal, NOT an isolated muzzle

This is the single most important thing to understand about matching quality.

Both `/register` and `/search` do the same thing (`main.py`
`_run_registration_pipeline` / `_run_search_pipeline`):

```
quality_check → crop_cattle(img) → quality_check_cv2(crop) → embed_batch([crop])
```

`crop_cattle` (`pipeline/yolo_crop.py`) is a **COCO YOLO (`yolov8s.pt`)** that
finds the **whole animal's bounding box** (classes `{17,18,19,20,21}` =
horse/sheep/cow/elephant/bear, to catch buffalo misclassifications). It is
**not** a muzzle detector. `pipeline/muzzle.py` is misleadingly named — it is
only the DINOv2 forward pass (`embed_batch`), no muzzle detection anywhere.

So the encoder (`GodhaarModel`, trained on tight 518×518 **muzzle** crops) is
fed whole-animal crops at inference. `pipeline/preprocess.py` then does a
**non-aspect-preserving resize to 518×518**. For a close-up muzzle photo the
cattle-body box ≈ the head/face, so it's muzzle-ish but still includes
face/coat/background — which two similar animals (e.g. two black cattle)
share. This narrows the score spread between different animals. **If matching
is weak, this is the primary structural reason — there is no server-side
muzzle isolation.**

### ⚠️ Fixed: the close-up "return the full frame" bug (this made EVERY cow match the one in the DB)

Reported live: with one animal registered, **every** search returned that
animal — a white cow came back as the registered black cow; and registering a
second, genuinely-different black cow was rejected as a **duplicate** of the
first. Confirmed from the live inference log: three different search photos,
same `top1_faiss_id` every time, one scoring only 0.61.

Root cause was one branch in `crop_cattle` (`pipeline/yolo_crop.py`, was
~line 298): if the detected box filled **≥ `CLOSE_UP_AREA_PCT = 0.55`** of the
frame, it returned the **entire uncropped image** (`return img, "FULL_IMAGE",
conf`) instead of a bbox crop, on the theory the animal was "too close for a
meaningful crop." That is backwards for this pipeline: the capture UI tells
officers to **fill the frame with the muzzle**, so a *correctly-taken* photo
routinely has the box covering >55% of the frame and hit this branch. The
field log showed it firing on **nearly every** register and search image. Net
effect: most photos were embedded as the full raw scene (dirt, rope, foliage,
other cattle), so different animals in similar settings collapsed to nearly
identical embeddings → everything matched everything.

**Fix (applied):** removed the close-up passthrough entirely; `crop_cattle`
now always returns the tight (padded) bbox crop, exactly as it already did for
non-close-up shots. Also removed the now-unused `CLOSE_UP_AREA_PCT` from
`godhaar/config.py`. `crop_cattle`'s only production callers are the muzzle
embedding path (`main.py` register/search) — there is **no** whole-body use
case that needed an uncropped fallback, so nothing else depends on the old
behaviour. The `FULL_IMAGE` status is still returned by the two legitimate
paths that keep it: `no_crop=True` (explicit caller opt-out) and
`FULL_IMAGE_NO_YOLO` (model not loaded) — both untouched.

Verified before finalising: all server modules import; both register and
search pipelines run end-to-end producing valid (3,256)/(256,) unit-norm
embeddings; `crop_cattle` runs without exception on close-up and normal
photos; 4 real non-close-up photos showed **identical** behaviour (no
regression), and the one over-threshold photo now crops instead of passing the
full frame through.

**Status: committed** (`1db86da`, on `feature/cctv-video-analytics` — this was
still an uncommitted working-tree change when this note was first written).
Not yet confirmed deployed — a running GPU server only picks this up once
someone pulls the branch and restarts the inference container. **This fix
removes the catastrophic case but has NOT been confirmed to fully fix
matching** — see "Still open" below.

### Still open: is the crop fix ENOUGH? (needs a real different-animal test)

The crop fix definitely removes the full-frame collapse. What it does **not**
prove is that two *genuinely different* animals now score below threshold —
because the only local test pair (`Karunya` vs `Rama`, the
`debug_crop_output/` and root `*.png` sets) turned out to be the **same
animal** photographed twice (both black, same white forehead blaze, same
halter/setting — visually confirmed). So the "cross-animal" scores of
0.76–0.86 measured there are actually **correct same-animal** scores, not
false matches. That test proves nothing about telling two cows apart.

**To actually confirm search/duplicate is fixed, test two genuinely different
animals** — ideally the reported field pairs (the registered black cow + the
white cow that wrongly matched; and the two different black cows that were
wrongly rejected as duplicates). Run each through the fixed pipeline and check
the different-animal cosine drops **below 0.72** (search) / **below 0.80**
(duplicate). If it does not, the fix needs more than the crop — see next.

### A muzzle detector CAN run server-side (verified) — the likely next lever

If the whole-animal crop still doesn't separate different animals, the real
fix is to isolate the **muzzle** before embedding, matching the encoder's
training distribution. Confirmed working locally: the **app's muzzle model**
(`best_float16.tflite`, a single-class YOLOv8n, class name `cattle_muzzle_3`,
from the Telangana app assets) loads and runs **through the existing
`ultralytics`** dependency — `YOLO(path, task="detect")` — which auto-pulls a
small `ai-edge-litert` backend. On a real photo it found the muzzle at 0.888
conf with a tight box. So a server-side muzzle crop is a low-friction change if
needed; wire it identically into register AND search so stored and query
embeddings stay in the same space. (Do NOT source detectors from
`neeraj_detection` — off-limits.)

## Duplicate detection (register) — same crop bug poisons it, and the color gate is weak

`main.py` `/register`: embeds the new animal's 3 muzzle photos, **averages
them into one embedding** (`embeddings_np.mean(axis=0)`), runs
`restricted_search` against the GPS-nearby candidates the API server supplied,
and rejects with **409** only if **BOTH**:
1. cosine `score ≥ DUPLICATE_THRESHOLD`, **and**
2. **color match**: `stored.body_color == new_body AND stored.muzzle_color == new_muzzle`.

Two gotchas this caused (two different black cows wrongly merged):
- **The color gate is near-useless for black cattle** (the common case): both
  cows are BLACK body + dark muzzle → same labels → `color_match = True`. The
  regression run showed muzzle color coming back `BLACK` at only 0.43
  confidence — the label is not discriminative.
- **Cosine ≥ threshold** was inflated by the same full-frame crop bug above.
- **Threshold discrepancy:** the `/register` docstring says `DUPLICATE_THRESHOLD
  (0.95)` but the real value in `godhaar/config.py` is **`0.80`** (comment:
  lowered from 0.95 to catch re-registrations under different lighting). 0.80
  is a low bar; with the crop bug removed, re-check whether genuinely different
  black cows still clear 0.80 — if so, this threshold (and/or the color gate)
  needs revisiting, not just the crop.

**Update 2026-08-10 — the flat AND-gate described above is gone.**
`feature/cctv-video-analytics`'s merge replaced it with a **tiered** verdict
in `/register`: at or above `DUPLICATE_HIGH_CONFIDENCE_THRESHOLD` the
embedding alone decides (color can no longer veto a genuinely correct
muzzle match — the exact "black-cattle color gate" failure mode above was
a live double-registration incident, not a hypothetical); below it, only
`muzzle_color` needs to corroborate, deliberately **not** `body_color`
(which comes from `detect_primary_animal()` and has no quality/multi-cattle
gate the way muzzle photos do). `BYPASS_QUALITY_GATES` and
`BYPASS_DUPLICATE_CHECK` (both `main.py`, added 2026-08-08/09 for the field
bulk-registration push) are back to `False` as of 2026-08-10 — sync is
done, both gates enforce for real again, now against this tiered verdict.
Reverting was a clean two-value flip: `_run_registration_pipeline`, where
`BYPASS_QUALITY_GATES` does its work, was untouched by the tiered-gate
change. Also added, same date: per-stage `time.monotonic()` logging
through `_run_registration_pipeline` (muzzle quality/crop gate,
`embed_batch`, body/muzzle color, morphology, total, plus per-slot
`crop_cattle` timing with which retry attempt fired) — purely additive,
no behavior change, added because registration-latency complaints had no
real breakdown to point at yet, only guesses.

**A LightGlue keypoint-matching POC exists but is NOT wired into
anything** (`experiments/lightglue_poc/match_muzzles.py`, gitignored,
uses `kornia`'s DISK+LightGlue installed ad hoc, not in `pyproject.toml`).
The idea: a second, independent signal alongside the embedding score —
does the query and candidate muzzle actually share local keypoint
structure — that could in principle guard the `DUPLICATE_HIGH_CONFIDENCE_THRESHOLD`
tier above. Only smoke-tested (proves the plumbing runs, not that it
helps) on two arbitrary unlabeled photos. Same rule as everywhere else in
this file: do not wire this into a real accept/reject decision without
validating on real different-animal photos first — an unvalidated second
signal is exactly the kind of "expensive step that seems obviously right"
this file's GrabCut story already warns about.

## ⚠️ Fixed: body color returned BLACK for white/brown animals (the "dominant color" was fabricated)

Reported live twice: a **white Ongole-type bull** classified `BLACK`, and a
**brown-and-white Gir cow** classified `BLACK` on *both* its front photos.
Root cause was **not** photo quality — it was two independent defects in
`wildlife/color/utils.py::extract_dominant_lab_features`, the helper shared by
`body_color.py` **and** `muzzle_color.py`:

1. **The "dominant color" was a histogram bin center, not a measurement.** It
   binned the a*/b* channels 16 wide and returned the **bin center** as
   `dominant_lab`. The neutral point (128) sits on a bin **edge**, so every
   near-neutral (grey/black/white) coat landed in the bin centered at
   `(+8, +8)` — a fabricated chroma of **11.3 regardless of the real color**.
   Verified on real photos: 4 different animals all returned the *identical*
   chromaticity `(+8.0, +8.0)`, and the two reported Gir photos returned
   byte-identical `dominant_lab = [32.94, 8.0, 8.0]`. Because 11.3 is
   permanently below `NEUTRAL_THRESHOLD` (15), **`BROWN` was literally
   unreachable for any animal.**
2. **Peak-finding ignored L\* entirely** — the histogram was 2D over a*/b*
   only. A black coat and a white coat are both achromatic, so they shared a
   bin and had their lightness values **median-ed together**; that merged
   lightness is what then decided BLACK vs WHITE. This is the direct mechanism
   for the reported bug: a white animal against dark ground/shadow gets its
   lightness dragged down and comes back BLACK.

**Compounding:** `get_body_roi()` cropped a fixed percentage of the **frame**
center, not the animal — so most sampled pixels could be background/shadow
even before the algorithm ran.

### What the fix actually required (three layers, each found by testing the previous one)

Replacing the histogram with k-means was **necessary but not sufficient**.
Each layer below was added only after measuring that the previous one still
got real photos wrong — worth knowing before "simplifying" any of it away:

1. **k-means over full L\*a\*b\*** (`_kmeans_lab`, k=4, k-means++ init, pure
   numpy — sklearn is not a dependency here). Returns each cluster's **real
   centroid**. Fixes both defects above. **Fixed RNG seed (42) is mandatory,
   not cosmetic:** `/register` 422s when two front photos disagree on body
   color, so an unseeded clusterer would fail registrations at random.
   Verified deterministic across repeated runs (label, LAB and confidence all
   identical).
2. **Animal localization** (`roi.py`). k-means alone
   still returned the *background's* color, because "largest cluster" **is**
   the background when the animal is a minority of the frame. `get_body_roi`
   now crops to a YOLO-detected animal box.
   Detection uses a **new `detect_primary_animal()`** in `pipeline/yolo_crop.py`,
   deliberately *not* `crop_cattle()`: that enforces `MAX_CATTLE_PER_IMAGE=1`,
   which is right for muzzle embedding but wrong here — a goshala photo
   routinely has other animals in frame, and that is no reason to refuse to
   read the subject's coat. It picks the **largest** box, not the
   highest-confidence one (the photographed animal is nearest the camera).
   When localization is unavailable (YOLO not loaded, `pipeline/` not
   importable, or no detection) it falls back to the old fixed center crop.
   A **center-weighted spatial prior** is applied either way, so a centered
   subject isn't outvoted by peripheral background.
3. **Aggregate cluster weights BY LABEL, not by cluster**
   (`aggregate_clusters_by_label`). Clustering over L\* means one perceptual
   color **fragments across several clusters** that differ only in lighting.
   Measured on the real Gir photo: the coat split into `BROWN 0.29` +
   `BROWN 0.27` while a white blaze formed a single `0.29` cluster — so
   "heaviest cluster" was a coin-flip the coat could *lose*. Summing per label
   first (BROWN 0.56 vs WHITE 0.29) asks the question that actually matters.

4. **`NEUTRAL_THRESHOLD` recalibrated 15.0 → 9.0 — this was the real root
   cause of the remaining errors, and nearly got misattributed to
   segmentation.** Measured cluster chroma on real photos:
   genuinely achromatic (white/grey Brahman) = **0.05–7.1**; brown Gir coat =
   **12.4–16.7**. The old 15.0 sat *inside the brown distribution*, so lit
   parts of a brown coat fell to `GREY` and shadowed parts to `BLACK`. 9.0
   sits in the empty gap between the two populations.

### GrabCut was added, then removed — read this before re-adding it

An intermediate version ran **GrabCut** foreground segmentation inside the
YOLO box, and it looked indispensable: without it, brown animals came back
`SPOTTED`/`BLACK` (5 of 7 real photos wrong). It was **misattribution**. The
real cause was the `NEUTRAL_THRESHOLD` miscalibration above; GrabCut was only
nudging cluster centroids across an arbitrary line.

Two measurements exposed it, both worth repeating on any similar "essential"
component:
- **Cost:** ~6700 ms/image, ~50x the rest of the pipeline. At two front photos
  per `/register` that is ~13 s of added request latency — it turned body
  color from 59 ms to 8121 ms, a **137x regression** that no accuracy gain
  would have justified.
- **Non-monotonic accuracy in resolution:** 6/7 correct at 384 px but **5/7 at
  512 px**, and 7/7 only at full resolution. A component whose accuracy is not
  monotonic in input quality is not doing the job it appears to be doing — it
  was landing on right answers by luck.

With the threshold calibrated, GrabCut changes **no label** on the real photo
set, so it was deleted rather than tuned. `get_body_roi()` consequently
returns a plain `np.ndarray` again (not `(crop, mask)`), and the `mask`
parameters were removed from `calculate_median_lab` /
`extract_dominant_lab_features` rather than left as dead paths.

**Lesson worth keeping:** the first fix that makes the numbers go green is not
necessarily the fix. Measure cost, and check that accuracy degrades *smoothly*
when you weaken the component — if it doesn't, you have found a coincidence,
not a cause.

**`SPOTTED`/`MIXED` detection was rebuilt on top of this**, replacing a
`grayscale std_dev > 25` kludge that fired just as readily on a solid-colored
animal in harsh sunlight. It now means "the runner-up **color** holds a
comparable share of the animal." `SPOTTED_RATIO_MIN`/`MIXED_RATIO_MIN` were
raised `0.25 → 0.60` for a measured reason: the same Gir cow gave white/brown
ratios of **0.51 and 0.22 on its two register photos**, purely because the
head fills more of one frame than the other. Any threshold between those two
values labels one photo `SPOTTED` and the other `BROWN` and **trips
/register's 422 on a perfectly good pair.**

### Verified against the real reported photos (before → after)

Across 8 real photos: **body 0/8 → 8/8, muzzle 4/8 → 8/8**, at 110 → 315
ms/image for both classifiers combined (the extra is `detect_primary_animal`
plus the muzzle detector; the 8-second GrabCut regression is gone).

| photo | old body | new body | truth |
|---|---|---|---|
| reported Gir front1/front2 | `BLACK` | **`BROWN`** | brown |
| white/grey Brahman calf ×2 | `BLACK` | **`GREY`** | white/grey |
| brown Gir calf ×2 | `GREY` | **`BROWN`** | brown |
| brown Gir wide shots ×2 | `GREY` | **`BROWN`** | brown |

Both photos of each animal now agree, so `/register`'s consistency check
passes. Verified deterministic across repeated runs (label + confidence). `wildlife/color/tests/test_color.py` gained a
`TestBodyColorRegressions` class — **7 of its tests fail against the old
implementation and all 10 pass against the new one**, confirmed by running the
new suite against a pristine copy of the pre-fix module. Note the tests build
scenes where the coat is a **minority** of the frame: a naive synthetic image
with the coat filling most of the frame passes even on the buggy code and
proves nothing.

## ⚠️ Fixed: muzzle color was reading the COAT, not the muzzle

Same report, second half: muzzle color came back `PINK` for obviously black
muzzles. Two defects, both distinct from the body-color bugs above (though it
also shared the fabricated-bin-center helper, fixed above):

1. **The ROI was never localized on the muzzle.** `main.py` passes
   `crop_cattle()`'s **whole-animal** box to `extract_muzzle()`, and
   `get_muzzle_roi()` took a fixed center crop *of that box* — i.e. the
   animal's **neck/chest**. It reported `PINK` because it was measuring brown
   hide. No threshold change can fix measuring the wrong pixels.
2. **`_classify_muzzle_lab()` returned `MIXED` as its catch-all for a SINGLE
   color sample.** `MIXED` describes a muzzle carrying two skin colors — a
   property of the whole muzzle; one cluster is one color by definition. A
   real muzzle whose lower lip fell outside both the BLACK and PINK rules
   contributed a phantom "MIXED color" that then outvoted the real reading and
   mislabeled a plainly black muzzle. It now returns only `BLACK`/`PINK`/
   `UNKNOWN`; `MIXED` is produced solely by the aggregation step when both
   real skin colors hold comparable share. `UNKNOWN` clusters get no vote on
   the color but **still count against confidence**, so a partly-unreadable
   muzzle doesn't report false certainty.

**The muzzle detector is now wired in** (`pipeline/muzzle_detect.py`), using
the app's `best_float16.tflite` (single-class YOLOv8n, `cattle_muzzle_3`) via
`YOLO(path, task="detect")`. Measured **0.82–0.91 confidence** on real field
photos with tight boxes. `get_muzzle_roi()` crops to that box (8% inset to
drop hair clipped at the edges) and falls back to the old fixed center crop
when the detector is unavailable or finds nothing. **No GrabCut here**, unlike
the body ROI — a tight muzzle box is nearly all skin, so there is no
background to segment and running it would only risk eating real nostril/lip
pixels.

It picks the **highest-confidence** box, deliberately the opposite of
`detect_primary_animal()`'s largest-box rule: there is exactly one muzzle on
the subject, and a background animal nearer the camera would win on size.

Deployment notes:
- `MUZZLE_MODEL_PATH` (`godhaar/config.py`, env-overridable) defaults to
  `appstorage/Models/muzzle_detect/best_float16.tflite`. **`appstorage/` is
  gitignored**, so the file must be placed on the deployment volume alongside
  the other models — it is NOT carried by a git pull.
- `ai-edge-litert==2.1.6` added to `pyproject.toml`. `ultralytics`
  auto-installs it on first use, but that needs network and a writable env,
  neither guaranteed in the container.
- Loading is **non-fatal**: a missing model logs a warning and degrades to the
  fixed crop rather than failing startup.
- `MIN_MUZZLE_ROI_WIDTH/HEIGHT = 64` (new) — the old 120px gate was sized for
  a fixed crop of a whole frame and **rejected real 117×113 detector crops as
  "too small,"** discarding the best pixels available in favour of nothing.

**Verified on real photos: 8/8 muzzles correct** (was `PINK`/`MIXED`), through
`main.py`'s exact path (`crop_cattle` → `extract_muzzle`).

### Still open

- **Only BLACK muzzles have been tested.** All 8 real photos available are
  black-muzzled, so this confirms the classifier stopped reporting `PINK` for
  black muzzles — it does **not** prove it can tell PINK from BLACK on a real
  pink muzzle. `L_BLACK_MAX=40` and the `L>45 and a>4 → PINK` rule are still
  the original uncalibrated guesses, **deliberately not blind-tuned** (same
  reasoning as the blur-threshold story elsewhere in this file). Needs a
  labeled PINK/BLACK set; synthetic tests cover both directions but synthetic
  is not calibration.
- **The muzzle ROI bug cannot be reproduced synthetically** — the detector
  won't fire on a synthetic image, so the localization half of this fix is
  covered only by real-photo verification, not by the unit suite.
- **Every animal registered before this has wrong stored `body_color` AND
  `muzzle_color`.** The duplicate gate (`main.py`, `stored.body_color ==
  new_body and stored.muzzle_color == new_muzzle`) compares new correct labels
  against old fabricated ones, so it will silently stop matching those rows
  until they are re-extracted. Needs a backfill on the go-apiserver side —
  not done here.
- `L_BLACK_MAX`/`L_WHITE_MIN` remain uncalibrated for **body** color too: the
  white calf reads `GREY` not `WHITE` because indoor shade puts it at L\*≈44,
  well under `L_WHITE_MIN=75`. Better than `BLACK`, still not calibrated.
- **Unrelated, found while testing:** real goshala front photos hit
  `RECAPTURE_MULTI_CATTLE` in `crop_cattle` (several animals in frame).
  ~~Pre-existing, untouched.~~ **Now fixed — see "Multi-cattle goshala photos"
  below.**

**The embedding path is deliberately unchanged.** `main.py` register/search
still feed `crop_cattle()`'s whole-animal crop to the encoder. Routing the
muzzle detector into embedding would likely improve matching (see the section
above on the encoder being trained on tight muzzle crops), but it moves stored
and query embeddings into a different space and **invalidates every vector
already in the FAISS index** — a separate, much larger migration.

## Horn/ear morphology — a new signal for the weak color gate, v1 is an unvalidated heuristic

> ⚠️ **The field names in this section (`horn_length_ratio`, `ear_span_ratio`)
> are HISTORICAL.** They were replaced with categorical `has_horns`/
> `horn_shape` fields — see "Redesigned: ratios → has_horns/horn_shape"
> below. The architecture, the "no dataset exists" finding, and the
> not-wired-into-any-decision policy described here are all still current;
> only the specific field names changed.

Requested directly in response to the "color gate is near-useless for black
cattle" problem above: return horn-length and ear-span as an extra signal,
the same way `body_color`/`muzzle_color` already are, in **both**
`/register` and `/search` — not gated on anything, just returned alongside
the existing colors so the caller (go-apiserver) can eventually factor it
into the duplicate/match decision the same way it already uses color.

**There is no horn/ear detector, keypoint model, or labeled dataset
anywhere in this project.** Checked both this repo's bundled `wildlife/`
copy and the full `Godhaar/Wildlife` source repo — neither has anything
beyond the existing color classifiers. So this had to be built from
scratch, and — critically — there was no ground truth to validate it
against (no equivalent of the blur-threshold calibration story elsewhere in
this file, where real photos were measured against the real server formula
before shipping). Absolute measurement (centimeters) was also ruled out
immediately: a phone photo carries no scale reference, so the same horn
would measure differently depending only on how far away the phone was
held.

**What got built** (`pipeline/morphology.py`), mirroring `pipeline/color.py`'s
exact architecture — a `MorphologyExtractor` ABC with a
`RuleBasedMorphologyExtractor` implementation, swappable later for a trained
model without touching callers:
1. Run `crop_cattle` (the existing whole-animal YOLO) on the front-facing
   photo to get a scale-normalized crop.
2. Take the top 35% of that crop as the head/horn/ear band — the same region
   `wildlife/color/roi.py` already excludes from body color for the opposite
   reason (its own comment: "often contains background, sky, ears, horns").
3. Canny-edge the band, take the largest contour, measure its top/left/right
   extremities.
4. Return everything as **ratios of the crop's own width** — `horn_length_ratio`,
   `ear_span_ratio` — never an absolute length, plus a `confidence` **capped
   at 0.6** so a heuristic with zero validation data can never report itself
   as more certain than a real classifier would.

Wired into both endpoints exactly like color: `RuleBasedMorphologyExtractor`
loads at startup (`app.state.morphology_extractor`, `dependency.py`'s
`get_morphology_extractor`), `MorphologyResult` added to
`RegisterResponse`/`SearchResponse` (`schema.py`) as `morphology`. Register
gets 2 front photos, so both are read and combined via a new
`average_readings()` helper — confidence-weighted, so a failed reading
(confidence 0) doesn't drag a good one toward zero, and the combined
confidence is the **mean**, not the max, so "only 1 of 2 photos worked"
correctly reads as less certain than "both worked." Unlike color, there's no
majority/consistency check across the 2 front photos — these are continuous
ratios, not categorical labels, so "the two readings disagree slightly"
isn't a retake-worthy error the way a color mismatch is.

**Smoke-tested against the real local `Karunya`/`Rama` front photos**
(no labeled ground truth exists, so this only confirms it runs and fails
open, not that the numbers are meaningful): 2 of 3 photos produced a
reading (confidence capped at 0.6 as designed), the third failed
detection and correctly returned the zero/unknown reading rather than
throwing. `ear_span_ratio` came back ~0.999 on both successful reads —
plausible for a wide horn/ear band, but also consistent with the contour
just tracing the crop's own edges rather than isolating ears specifically;
this is exactly the kind of thing that needs real validation, not
guessed at.

**Deliberately NOT wired into any accept/reject decision** — not the
`/register` 409 duplicate check, not any match scoring. It's return-only,
same as this section's opening paragraph said, until someone validates it
against real different-animal photos the way the crop fix above still
needs to be. Wiring an unvalidated heuristic into a decision that can block
a real farmer's registration would repeat the exact mistake this file's
blur-threshold story warns against: shipping a threshold before checking it
against real data. **Still open, same as the crop fix above:** get photos
of two genuinely different animals (ideally with visibly different
horn/ear shapes) and check whether `horn_length_ratio`/`ear_span_ratio`
actually separate them before trusting this for anything beyond display.

### Silent zeros were replaced with an explicit `status`/`reason`

First cut of this returned `{horn_length_ratio: 0.0, ear_span_ratio: 0.0,
confidence: 0.0}` on any failure — indistinguishable from a real
low-confidence reading, and gave a caller nothing to explain WHY. Fixed:
every reading now carries `status` (`OK`, `INVALID_IMAGE`,
`NO_ANIMAL_DETECTED`, `NO_CLEAR_SILHOUETTE`, or register-only `PARTIAL` —
one of two front photos failed) and a human-readable `reason` string. Only
`OK`/`PARTIAL` carry a real (possibly confidence-weighted) reading —
anything else means the ratios are the zero default and must not be read
as data. `average_readings()` (register's 2-photo combine) propagates this
too: if only one of two photos produced a reading, status is `PARTIAL` with
a reason naming how many failed, not a silently-blended number that looks
as trustworthy as two genuine agreeing readings.

**What "no horn visible" actually means — asked directly, answered
honestly:** if a horn points backward, is occluded, or the animal is
genuinely polled/dehorned (routine in Indian cattle, not an edge case),
this v1 heuristic cannot tell those apart. All of them come back as
`NO_CLEAR_SILHOUETTE` or a low `horn_length_ratio` with `OK` status — a
weak/absent contour above the head band looks identical whether the cause
is "no horns," "horns not visible from this angle," or "bad lighting on
the crown." **This is a structural limit of single-2D-front-photo
silhouette analysis, not a bug to chase** — no amount of tuning
`MIN_CONTOUR_AREA_FRACTION` or the edge-detection parameters fixes it,
because the information (a backward horn's true shape) simply isn't in a
front-facing 2D photo. The two real fixes, neither in scope here: a
horn-specific detector trained to recognize "no horn present" vs "horn
occluded" as different classes (needs labeled data that doesn't exist —
see above), or a second capture angle (side profile) where a backward horn
becomes visible — a capture-flow product decision for the app, not
something this service can solve alone. **Until then, the contract is:**
treat any non-`OK` status, and any `OK` reading with a near-zero
`horn_length_ratio`, as "no horn confirmed in this photo" — never as
"confirmed no horns." Don't use a low reading as negative evidence
anywhere (e.g. don't let it argue two animals are "different" just because
one photo happened to hide the horn).

### `/search` candidates now carry stored morphology too — a real BREAKING CHANGE to its request contract

Follow-up ask: `/search`'s candidates should carry morphology the same way
`/register`'s already do, "so they can be compared against the query's
morphology — not just returned for the query animal alone." Before this,
`/search` only ever returned the QUERY's own extracted color/morphology;
each ranked match carried nothing but `faiss_id/score/rank/gap` — no stored
color or morphology for the animal that was actually matched, even though
go-apiserver's `h.search()` already pulls `body_color`/`muzzle_color` per
candidate from its own DB (`FindFAISSCandidates`) — it just wasn't sending
that data into this endpoint the way `/register` does.

**Explicitly scoped to this service only, on the user's direct instruction
— go-apiserver is the CTO's side and was not touched.** That makes this a
real, not theoretical, breaking change: go-apiserver's `Search()` client
(`internal/inference/client.go`) currently sends bare repeated
`candidate_ids` form fields; this endpoint no longer accepts that shape at
all. **Until go-apiserver's Search client is updated to send a `candidates`
field with the same JSON shape `/register` already uses, every `/search`
call will 422.** This is the main challenge worth flagging: this commit
alone does not ship a working `/search` — it ships one half of a two-repo
change, deliberately, because the other half isn't this repo's to make.

What changed here (`main.py`, `schema.py`):
- `/search`'s `candidate_ids: list[int] = Form(...)` → `candidates:
  Form(...)` — same `CandidateInfo` list `/register` already parses, now
  with `horn_length_ratio`/`ear_span_ratio`/`morphology_confidence` added
  as **optional** fields (default `None`, not a fabricated `0.0`) — so a
  caller with no morphology to send yet (nothing persists it — see below)
  doesn't have to send anything new to keep working, once it's updated to
  the new `candidates` shape at all.
- `MatchCandidate` gained the same optional fields plus `body_color`/
  `muzzle_color`, populated by echoing back whatever the matching
  `CandidateInfo` in the request carried. **No comparison/similarity math
  is computed here** — this endpoint hands back both sides (the query's own
  `morphology` at the top level, each match's stored morphology on
  `top_matches[i]`) and leaves any actual comparison to the caller,
  consistent with the "return-only" decision above and with the fact that
  `decide()` (wherever it lives now) is explicitly not this repo's to
  change.

**Second challenge, worth knowing before anyone wires this further:**
go-apiserver's own code already explains why color isn't used to filter
search results — `h.search()`'s comment: *"Color labels from inference are
intentionally NOT used to hard-filter — classifier confidence is
unreliable."* That decision was made for **color**, which has actual
calibration behind it. This morphology heuristic has none at all (see
above — zero labeled data, capped confidence, `ear_span_ratio` reading
suspiciously close to 1.0 on real test photos). So even once go-apiserver
is updated to plumb this through, the same reasoning that kept color out of
`decide()` applies at least as strongly to morphology — there's no basis
yet to trust it more than the thing that was already rejected for the same
job.

**Third challenge, already true before this change and unaffected by it:**
nothing persists horn/ear data anywhere. `animal.CandidateRow` (go-apiserver)
only has `body_color`/`muzzle_color` columns; there's no
`horn_length_ratio`/`ear_span_ratio` column, no migration, and `/register`'s
response `morphology` field isn't written into `CreateAnimalTx.Animal`
anywhere. So even with `/search`'s contract fixed on the go-apiserver side,
every candidate's morphology fields would come back `None` until a DB
migration + the register write path are also updated — again, not done
here, on the same "CTO's side" instruction.

### Redesigned: ratios → `has_horns`/`horn_shape` — explicit request to drop the numbers

On direct instruction: replace `horn_length_ratio`/`ear_span_ratio` with
"does the cattle have horns or not, if yes what's the shape (hardcoded
string/enum), if no horns it should be null" — categorical fields, the
same style `body_color`/`muzzle_color` already use, not a measurement.
Ear-specific fields were dropped entirely, not just renamed — the request
only asked about horn presence/shape.

`pipeline/morphology.py` now returns `has_horns: bool | None` and
`horn_shape: str | None` (`pipeline.morphology.HornShape` — `STRAIGHT`,
`CURVED`, `UNKNOWN`), replacing both ratio fields everywhere they appeared
(`MorphologyResult`, `CandidateInfo`, `MatchCandidate` in `schema.py`; the
`/search` match-echoing code in `main.py`). Detection reuses the same
pipeline as before (`crop_cattle` → head band → largest edge contour),
just interpreted differently:
- **Presence**: a protrusion above the head-band base ≥
  `MIN_HORN_PX_FRACTION` (3% of crop width) → `has_horns=True`; below that
  → `has_horns=False`, `horn_shape=None`.
- **Shape**: for a present horn, fit a line (`cv2.fitLine`) to the
  contour's upper points and measure normalized RMS deviation from it
  (`_classify_shape`) — below `STRAIGHTNESS_THRESHOLD` (0.10) → `STRAIGHT`,
  above → `CURVED`.

**`HornShape` deliberately has no `NONE` member — caught in review.** The
first cut had one (`horn_shape="NONE"` alongside `has_horns=False`), and
it was flagged as redundant: `has_horns=False` already says there's no
horn, so a `NONE` shape said the identical thing a second way, and a
second, different way (`horn_shape=None`, the JSON null) already existed
for "no reading at all" (non-OK status). Two representations of "nothing
here" — one a string, one a null — for two different reasons was exactly
the kind of confusion this file's own conventions try to avoid. Fixed:
`horn_shape` is `None` (not a string) whenever there's no horn to
describe, for either reason (`has_horns=False`, or status isn't OK).
`HornShape` now only has real shapes — `STRAIGHT`, `CURVED`, and
`UNKNOWN` (reserved for "a horn was found but its shape couldn't be
classified," e.g. too few contour points — this is the only case where
`has_horns=True` and `horn_shape` isn't a real shape).

**Shape classification is a rougher guess than presence detection was,
and presence detection was already unvalidated.** Both thresholds
(`MIN_HORN_PX_FRACTION`, `STRAIGHTNESS_THRESHOLD`) are picked, not
calibrated — there's still no labeled data anywhere in this project to
check them against (see the section above). `STRAIGHT` vs `CURVED` also
can't distinguish curl/spiral shapes from a simple bend — collapsed
into one `CURVED` bucket deliberately, rather than inventing more
categories (`CURLED`, direction-of-curve, etc.) that would need their own
uncalibrated thresholds on top of an already-uncalibrated one.

**`average_readings()` (register's 2-photo combine) had to change shape,
not just field names** — the old version could confidence-weight-blend
two numbers; there's no such thing as "the average of STRAIGHT and
CURVED." New behavior: both photos agree → that value, confidence =
mean; only one produced a reading → that one, `status=PARTIAL` (as
before); **both produced a reading but disagree → `status=INCONSISTENT`,
`has_horns`/`horn_shape` both `None`** — a new status, because picking
one of two disagreeing readings arbitrarily would misrepresent the one
not picked, which is worse than admitting uncertainty.

**Smoke-tested against the real `Karunya`/`Rama` photos** (still no
ground truth, so this only confirms behavior, not accuracy):
`Karunya front1` → `has_horns=True, horn_shape=STRAIGHT`; `Rama front1` →
`has_horns=True, horn_shape=CURVED` (straightness 0.101, barely over the
0.10 cutoff — a reminder this boundary is a guess, not a calibrated line);
`Karunya front2` → still correctly `NO_CLEAR_SILHOUETTE`, not miscast as
"no horns." Averaging `Karunya front1` (STRAIGHT) with `Rama front1`
(CURVED) correctly produced `INCONSISTENT` rather than picking one.

Still return-only, still not wired into any accept/reject decision, still
scoped to this repo only — none of the policy from the sections above
changed, only the field shape.

## ⚠️ Fixed: multi-cattle goshala photos — pick the dominant subject instead of rejecting

Reported live: registration was impossible at a goshala because the cattle
stand shoulder to shoulder, so a **correctly-framed** photo of one animal
always has neighbours in frame and `crop_cattle` returned
`RECAPTURE_MULTI_CATTLE`. There is no retake that fixes this — the officer
cannot make the neighbouring cow leave. Requested behaviour: *"the cattle
which is concentrated more should [be] given priority."*

**Fix:** when more than `MAX_CATTLE_PER_IMAGE` boxes survive dedup,
`crop_cattle` now calls a new `select_dominant_box()` instead of rejecting
outright. Each box scores

```
area_fraction * (1 - center_distance) ** SUBJECT_CENTER_WEIGHT_POWER
```

(`center_distance` = box center's distance from frame center, normalized by
the half-diagonal → 0 at dead center, 1 in a corner, so it is resolution- and
aspect-independent). The top box wins only if it beats the runner-up by
`SUBJECT_DOMINANCE_RATIO`; otherwise the image is still rejected as
multi-cattle.

**Why area alone was not enough — this is the measurement that shaped it.**
`godhaar/config.py` already carried a `DOMINANT_AREA_RATIO = 3.0` with the
comment "if top box is >=3x larger than next, drop smaller boxes" — **dead
code, nothing ever imported it.** Measured on the 5 reported photos, the
subject's *area* ratio over the neighbour was only **1.62x** (front1) and
**2.28x** (the muzzle shots), so a 3.0 area gate would have rejected 3 of 5.
Centrality is what actually separates them: the neighbour sits far off-center
in every photo (center distance 0.39–0.55 vs the subject's 0.04–0.16). With
the center term squared, the same photos score **5.9x, 7.2x and 42x** — a wide
gap that 2.0 clears with margin. `DOMINANT_AREA_RATIO` was deleted rather than
left dead.

**Confidence is deliberately NOT part of the score.** It measures how sure
YOLO is that something is a cow, not which cow was photographed — and on the
reported front photo the *background* white cow scored **higher** confidence
than the subject (0.922 vs 0.861), being unblurred and side-on. Scoring on
confidence would have picked exactly the wrong animal.

**The ambiguity fallback is kept on purpose.** When two animals are comparably
large *and* comparably centered, `select_dominant_box` returns `None` and the
422 still fires. That is the case the single-animal gate genuinely exists for:
an embedding that could belong to either animal is worse than a retake, since
it silently poisons the FAISS index. Verified with a synthetic equal pair
(two identical boxes mirrored about center) → still rejected.

**`detect_primary_animal()` was switched to the same score**, replacing its
"largest box" rule. This is a consistency fix, not cosmetics: front photos
reach *body color* through `detect_primary_animal` while muzzle photos reach
the *encoder* through `crop_cattle`, so under two different rules `/register`
could store the **neighbour's coat color against the subject's embedding**. It
keeps its old always-return-something semantics (no ratio gate) — the fallback
there is a fixed center crop of the whole frame, which is strictly worse than
the top-scoring animal's box even on a close call. Verified the old and new
rules pick the **identical** box on both reported front photos, so this
carries no regression risk for the body-color work validated earlier.

**Verified on the 5 reported photos:** all 5 now return `OK` (were
`RECAPTURE_MULTI_CATTLE`); the saved crops confirm the **black bull** was
selected and the white neighbour excluded in every one. Muzzle color reads
`BLACK` 3/3 (majority gate passes). The 14 existing color unit tests still
pass.

### Two MORE gates were blocking the same photos — a stack of three

Registration was blocked by three independent gates in series; fixing only the
first would have looked like no progress at all from the field. Each was found
by fixing the one in front of it and re-running the real
`_run_registration_pipeline`. **All three are now fixed and the reported
photos register end-to-end.** The other two:

#### 2. Body-color unanimity was a policy bug, not a calibration bug

`/register` 422'd whenever the 2 front photos read different body colors.
Fixed in `main.py::_resolve_disagreeing_body_colors`.

The tell is an asymmetry in the same function: **muzzle** color takes 3
samples and accepts a **majority**, while **body** color took 2 and demanded
**unanimity**. Body was never stricter because it is more reliable — it is
stricter only because you cannot form a majority out of 2. Meanwhile
go-apiserver deliberately does *not* hard-filter search on these same labels
("classifier confidence is unreliable"), so a signal too weak to filter a
search result was strong enough to block a registration outright.

Worse, the officer could not comply: the disagreement's usual cause is the two
front shots framing the animal differently, so the coat's share of sampled
pixels shifts and a near-boundary coat lands on either side. Retaking the same
two angles reproduces it exactly.

Now: the reading claiming at least `BODY_COLOR_MAJORITY_CONFIDENCE` (0.50) of
the sampled coat wins. That is **not a tuned number** — `body_color.py`
defines confidence as the winning label's *share of the sampled coat*, so 0.50
is literally "this color covers most of the animal," the same majority idea
muzzle color already uses. Still 422 when **both** readings are confident and
contradictory (a real retake case — possibly two different animals), or when
**neither** is decisive. Accepted readings have confidence **halved**,
following `average_readings()`'s existing convention for morphology's
`PARTIAL`. **Zero regression risk on the 8-photo body-color set: those photos
all AGREED, and the agreement path is untouched** — this code only runs where
a hard 422 previously fired.

#### 3. The blur gate was measuring the wrong pixels on a sharp photo

`quality_check_cv2` rejected a reported muzzle photo as `bad_quality
blur=17.50`. The photo is not blurry: **the same photo's full frame scores
511.97.** Blur is measured on the **central 50%** (`_BLUR_CENTER_FRAC`), a
guard against penalizing bokeh backgrounds — which assumes the subject is
central and the out-of-focus part peripheral. That assumption **inverts for a
tight YOLO crop**: the background is already cropped away, and dead-center is
now the animal's smooth hide (the bridge of the nose), while the texture that
proves focus — the muzzle's bead pattern, hair boundaries — sits off-center.
Note this became reachable for *every* close-up once the close-up passthrough
was removed (see the top of this file); it is not goshala-specific.

This is the same class of defect as the muzzle-color ROI bug documented above,
and the same rule applies: **no `BLUR_THRESHOLD` change can fix measuring the
wrong pixels** — lowering 20.0 to admit this photo would admit genuinely
blurry ones too. So the threshold was NOT touched. `_blur_score()` now takes
the **max** of the central region and the whole crop, answering the question
the gate actually cares about ("is the subject in focus anywhere?"). Being a
max, it is **monotonically ≥ the old value, so it cannot reject any image that
passes today** — verified over 200 random images. `quality_check()` (raw full
frame) keeps the central-50% rule, where the bokeh rationale genuinely holds.

#### Verified end-to-end

`main.py`'s real `_run_registration_pipeline`, the reported photos, encoder
stubbed (no embedding code was touched): **PASSED** — 3×256 unit-norm
embeddings, `body_color=BLACK` (conf 0.41, `RESOLVED_DISAGREEMENT`),
`muzzle_color=BLACK` (conf 0.87), morphology `INCONSISTENT` (return-only,
never a gate). 14 color unit tests still pass.

**Note on the sample photos:** two of the three supplied muzzle images were
**byte-identical duplicates** of each other. Registration expects 3 distinct
muzzle photos, so real captures should not look like this — it does not affect
the fixes, but it means the 3-photo majority vote was effectively a 2-photo
one here.

### Still open: the underlying body-color classifier is still wrong here

The gate no longer blocks registration, but the classifier that caused the
disagreement was **not** fixed — it is now tolerated, not corrected. The
measured cause:

| | front1 | front2 |
|---|---|---|
| top cluster chroma | **10.23** → `BROWN` | 3.66 → `BLACK` |
| runner-up/primary | **0.888** → `SPOTTED` | 0.219 → `BLACK` |

Two documented-as-uncalibrated constants are both implicated, and this animal
is **new evidence against the calibration story recorded above**:
- `NEUTRAL_THRESHOLD = 9.0` was set because measured achromatic coats ran
  0.05–7.1 and brown coats 12.4–16.7, with 9.0 "in the empty gap between the
  two populations." **This bull's coat sits at 10.23 — inside that supposedly
  empty gap.** Part of its dark coat therefore reads `BROWN` and part `BLACK`.
- `SPOTTED_RATIO_MIN = 0.60` repeats the exact failure its own note warns
  about: the same animal gives 0.888 and 0.219 on its two front photos purely
  because the head fills more of one frame than the other, and 0.60 sits
  between them.

**Deliberately NOT blind-tuned.** Moving either constant to make this one
animal pass would be the precise mistake this file's blur-threshold and
GrabCut stories warn against, and the 8-photo real set those values were
calibrated against **is no longer in the repo**, so a change cannot be checked
for regressions. Needs that photo set (or a new labeled one) before either
threshold moves.

### Still open: `SUBJECT_DOMINANCE_RATIO` (2.0) doesn't clear on all real photos

Found while investigating an unrelated question (a LightGlue keypoint-matching
experiment for search false-positives, see the Telangana frontend repo's
CLAUDE.md) that needed clean muzzle crops for 9 real registered animals from a
Dehradun goshala. Running those photos through the *current, already-fixed*
`crop_cattle` surfaced two that still hit `RECAPTURE_MULTI_CATTLE` today:

- **`UKDEJS189273/muzzle3.jpg`** — two real animals, comparably sized and
  centered: dominance scores **0.4458** vs **0.2342**, ratio **1.90** (needs
  ≥2.0 per `SUBJECT_DOMINANCE_RATIO`). Reproducible every run — both
  detections are comfortably above `YOLO_CONF` (0.568, 0.500), so this is a
  genuine near-miss on the ratio itself, not a detection-threshold flicker.
- **`UKDEJS852420/muzzle2.jpg`** — same failure mode, ratio **1.18**
  (0.2998 vs 0.2536) *when the second animal's box clears `YOLO_CONF=0.30`
  at all* (observed 0.330 in a full-batch run, alongside 39 other images).
  Run in isolation (fresh process, this photo only), the same second
  detection scored **0.285** — just under the 0.30 cutoff — and the image
  failed as `RECAPTURE_NO_DETECTION` instead of `RECAPTURE_MULTI_CATTLE`.
  Repeated 3x in isolation, always 0.285/`RECAPTURE_NO_DETECTION`; repeated
  as part of the full 40-image batch, always 0.330/`RECAPTURE_MULTI_CATTLE`.
  So **this specific photo's YOLO confidence is not perfectly reproducible
  across execution contexts** — something about batching position, prior GPU
  state, or CUDA algorithm selection shifts a borderline (~0.28–0.33)
  detection across the 0.30 accept boundary. The end result is the same
  either way (rejected, no usable crop), but which rejection reason fires is
  not stable, which matters if anything downstream ever branches on the
  specific status string.

Both are real, current-production field photos — not the synthetic
equal-pair test the original fix was verified against, and not the 5 photos
`SUBJECT_DOMINANCE_RATIO=2.0` was calibrated on (documented above: those
scored 5.9x–42x, nowhere near this boundary). So the goshala-crowding problem
the original fix targeted is **not fully solved**: photos exist today where
two animals are close enough in size/centering that registration or search
would still hit `RECAPTURE_MULTI_CATTLE` (or a flickering
`RECAPTURE_NO_DETECTION`) with no retake that fixes it — the exact complaint
the original fix exists to resolve, just at a harder margin than the
calibration set covered.

**Downstream effect worth flagging, not just the rejection itself:** in a
strictly-gated `/register` or `/search` call this photo 422s and blocks that
slot outright — visible, not silent. But `_run_registration_pipeline`'s
`BYPASS_QUALITY_GATES` fallback for a detection failure is the **raw,
uncropped frame** (background, other cattle, and all) going straight into the
DINOv2 embedding (`main.py`, the `if crop is None: ... cropped_images[i-1] =
img_bgr` branch). If that flag is ever active again during a bulk-capture
push, a multi-cattle photo like these two would silently embed a
diluted, multi-animal frame instead of failing loudly — worse than the
rejection this fix was built to avoid. (`/search`'s pipeline has no such
fallback at any time — see the flag's own history — so this specific risk is
`/register`-only.)

**Not fixed here — flagged for its own investigation, per explicit
instruction not to touch `crop_cattle` while this was still being
diagnosed.** Options worth considering later, none decided: lowering
`SUBJECT_DOMINANCE_RATIO` (real headroom may exist — the original calibration
margin was 5.9x+, far above 2.0 — but re-verify against the original 5-photo
set first so this doesn't regress the case the threshold was built for);
making the ambiguous-multi-cattle fallback smarter than "use the whole raw
frame" (e.g. crop to the union of the competing boxes, or to the
higher-scoring one anyway, rather than returning `None`); investigating the
run-to-run confidence non-determinism near `YOLO_CONF` directly (fixed seed?
disabled cuDNN autotune?); or simply treating both as genuine retake cases
and improving the officer-facing message. Needs a larger real-photo sample
before any of these are decided.

## Registration quality gate now checks all 3 muzzle photos before failing, not just the first

`_run_registration_pipeline`'s quality/detection loop used to `raise` the
instant it hit the first bad muzzle image (bad blur, no cattle detected, or
bad crop quality) — the other two images were never evaluated at all. From
the app's side this meant a field officer retaking one bad photo,
resubmitting, and discovering a *second* bad photo only on the next round
trip — one full network request per bad slot. Flagged from the frontend side
(see the Telangana app's `CLAUDE.md` blur-fix notes, same underlying
`/register` endpoint) as a real field inconvenience worth fixing at the
source rather than patching around client-side.

Fixed by merging the quality-gate and crop/detection loops into a single pass
over all 3 images (`main.py`): each image still stops at its own first
problem, in the same priority order as before (quality → detection →
crop-quality), but a bad image no longer aborts evaluation of the remaining
two. Only after all 3 are checked does the endpoint raise `422`, with every
bad slot joined into one `detail` string — e.g. `muzzle_1: bad_quality
blur=8.9; muzzle_3: RECAPTURE_NO_DETECTION` instead of stopping at
`muzzle_1` alone. One retake cycle can now fix every flagged photo instead
of one at a time. Behavior is unchanged when all 3 images pass.

**Now structured** — see the next section. The combined `detail` string is
still produced (as `detail.message`, for logs), but the per-photo list it was
standing in for is now sent alongside it as `detail.failures`, so the app no
longer has to parse prose to learn which slot to retake.

Note the earlier claim here — that "go-apiserver relays this `detail` string
to the app as-is" — was never true. go-apiserver deliberately never renders
upstream prose into a user-facing message (`internal/inference/errors.go`:
"there is no field holding a message intended for display"), and a 4xx body
without an `error_code` was classified there as a *contract* failure and shown
to the officer as a generic "our team has been notified". Every 422 in this
loop, and every 409 duplicate, was being discarded at that boundary.

## Error envelope: every 4xx verdict carries an `error_code`

Deliberate rejections answer with

```json
{"error_code": "IMAGE_TOO_BLURRY", "detail": {...}}
```

not FastAPI's bare `{"detail": ...}`. This is a hard requirement, not a
nicety: go-apiserver keys its entire user-facing response off `error_code`,
and a body without one never reaches the officer as a real verdict (see
above). `schema.ErrorCode` and `domainCodes` in go-apiserver's
`internal/inference/errors.go` are **one contract and must change together**.

Built in `errors.py`. Raise via its constructors, never by hand:

| Situation | Code | `detail` shape |
|---|---|---|
| Duplicate muzzle (409) | `DUPLICATE_ANIMAL` | `matched_faiss_id`, `top_score`, `body_color`, `muzzle_color` |
| Bad photo(s) (422) | `IMAGE_TOO_BLURRY` / `IMAGE_BAD_EXPOSURE` / `IMAGE_TOO_SMALL` / `IMAGE_UNREADABLE` / `NO_ANIMAL_DETECTED` | `message` + `failures[]` of `{slot, stage, error_code, reason}` |
| Several animals, no clear subject (422) | `MULTI_CATTLE` | same |
| Several photos failing for *different* reasons (422) | `POOR_IMAGE_QUALITY` | same; per-photo codes survive on each entry |
| Front photos disagree on body colour (422) | `BODY_COLOR_INCONSISTENT` | same + `readings[]`, slots `front_1`/`front_2` |
| No muzzle-colour majority (422) | `MUZZLE_COLOR_INCONSISTENT` | same + `readings[]`, slots `muzzle_1..3` |

`failures[]` names every bad slot in one response, so one retake cycle can fix
all of them. go-apiserver attaches a short per-photo caption to each entry
(`perImageMessages`) — necessary because under the `POOR_IMAGE_QUALITY`
umbrella the envelope's own message describes the set, not any one thumbnail.

`readings[]` (`{slot, label, confidence}`, colour codes only) is the reason
the colour verdicts are worth returning at all. "Your two photos disagree"
invites the officer to retake the same two photos of the same two animals and
get the identical rejection; "front_1 read BLACK, front_2 read WHITE" points at
the actual likely cause. It reaches the app as `details.color_readings`.

**What must NOT get an envelope**, and why the handler is opt-in rather than
blanket: failures that are not a verdict about this animal or these photos —
FastAPI request validation, wrong image count, malformed `candidates` JSON,
FAISS errors — stay plain `HTTPException`. Those genuinely *are* the two
services being out of step, and go-apiserver is right to classify them as
contract/transport faults. Stamping a code onto them would surface a renamed
form field to a farmer as "retake your photos".

**`MULTI_CATTLE` is separate from `NO_ANIMAL_DETECTED` on purpose.** They need
opposite instructions — get more of the animal in frame vs. get less of
everything else in it — and in a goshala, where cattle stand shoulder to
shoulder, multi-cattle is the *common* failure. Collapsing the two (which this
briefly did, while go-apiserver had no code for it) hands the officer the one
instruction guaranteed to make the next photo worse. `crop_cattle` only returns
`RECAPTURE_MULTI_CATTLE` after `select_dominant_box` fails to find a subject,
so it already means "two equally prominent animals", not merely "more than one".

Changing a code, or adding one, is a **two-repo change**: `schema.ErrorCode`
here, plus `domainCodes`, `domainResponses` and `perImageMessages` in
go-apiserver. A code missing from those last two reaches the app with no copy
attached and renders blank. go-apiserver has a test asserting the two tables
agree; there is nothing checking them against this enum, so that direction is
still manual. Deploy order does not matter — an unknown code degrades to a
contract fault rather than being guessed at, so the verdict is lost but never
misreported.

## Decision thresholds live in the API SERVER, not here

This service returns raw scores only. MATCH/REVIEW/UNKNOWN and the 3 km GPS
candidate filter are the API server's job (`decide()`: MATCH ≥ 0.82 with
rank1−rank2 gap ≥ 0.08; REVIEW ≥ 0.72; else UNKNOWN). A single candidate in
range means gap = 0, so it can never be a clean MATCH — a score ≥ 0.72 becomes
REVIEW, which the app still surfaces as a result.

## `/search` fusion tiebreaker: LightGlue as an additive, inert signal

Built on the validation in the LightGlue POC section above (13 real Dehradun
`/search` false-positive incidents, re-run against production crops — see the
Telangana frontend repo's CLAUDE.md for the full experiment). This wires that
validated signal into `/search` for real, as three new response fields that
**nothing reads yet**.

**What it does.** After `/search` computes its normal ranked `matches` (no
change to that computation), if the top-1 candidate is ambiguous —
`REVIEW_THRESHOLD ≤ score ≤ MATCH_THRESHOLD`, OR `gap < GAP_THRESHOLD` — and a
cached crop exists for that candidate, it runs DISK+LightGlue keypoint
matching between the query's own muzzle crop and the candidate's cached crop,
and attaches the result:

```
lightglue_checked: bool                                   # did the tiebreaker actually run
lightglue_num_matches: int | None                          # raw LightGlue match count
lightglue_zone: "likely_same" | "likely_different" | "ambiguous" | None
```

**Why this is safe to ship with zero go-apiserver changes.** Confirmed by
reading `go-apiserver/internal/inference/client.go:155` — it decodes this
server's JSON with plain `json.Unmarshal`, no `DisallowUnknownFields()`, so Go
silently ignores fields its `SearchResponse`/`SearchMatch` structs don't
declare. The three fields above are live and totally inert until someone adds
matching struct fields on the Go side and chooses to read them. Fields are
top-level on `SearchResponse`, not on `MatchCandidate` — they describe "query
vs. top_matches[0]" specifically, not a property of any one ranked row.

**This server still makes no decision.** `_SEARCH_MATCH_THRESHOLD_MIRROR` /
`_SEARCH_REVIEW_THRESHOLD_MIRROR` / `_SEARCH_GAP_THRESHOLD_MIRROR` (`main.py`,
values 0.82/0.72/0.08) are a hand-maintained **mirror** of go-apiserver's
`decision.go`, used only to decide whether it's worth spending GPU time on an
optional signal — not a copy of business logic living here (the rule in
`godhaar/config.py`'s docstring against that still holds; these three
constants deliberately live in `main.py`, not there). They're also applied to
this server's own raw per-embedding `top_matches` ranking, computed *before*
go-apiserver aggregates multiple embeddings per animal and applies its
attribute-agreement adjustment — an approximation of what go-apiserver will
decide, not identical to it. There is no shared source between the two repos;
if `decision.go`'s thresholds move, these need updating by hand.

**Where the query and candidate crops come from.**
- Query crop: free — `_run_search_pipeline` already computes it for the
  embedding, now also returns it.
- Candidate crop: `pipeline/muzzle_crop_cache.py`, a local file cache at
  `MUZZLE_CROP_CACHE_DIR` (default `/appstorage/muzzle_crops/<faiss_id>.jpg`
  — the same host-mounted volume `FAISS_INDEX_PATH` already lives on, so no
  new infrastructure). Written by `/register` right after `faiss_ids` are
  assigned, keyed by `faiss_id` — which is 1:1 with a specific muzzle crop,
  so the cache always holds exactly the crop that produced whichever
  embedding scored highest, no "which of the 3 photos" ambiguity.
  **This was the deciding factor over fetching from GCS at search time**:
  inference_server has no GCS credentials and no DB connection to resolve
  `faiss_id → image_key` (that mapping lives in go-apiserver's Postgres
  `embeddings`+`images` tables) — either GCS route would have meant a
  go-apiserver change or a new, larger coupling. The local cache needs
  neither.
- **Known gap, accepted deliberately: no backfill.** Only animals registered
  *after* this shipped have a cached crop. Every animal already in the FAISS
  index (139 Dehradun animals as of this writing) gets `lightglue_checked:
  false` on an ambiguous search until re-registered. A one-time backfill
  (fetch each existing animal's muzzle image from GCS once, with temporary
  credentials, populate the cache) is a real but bounded task — not
  attempted here, not needed to ship this.

**Zone thresholds — derived from data, not guessed.** From
`lightglue_fp_results_cropped.csv` (the production-crop LightGlue
validation), restricted to the 48 pairs whose *both* images actually cleared
production's quality/detection gates — the only population a live `/search`
call can ever present this function with (a gate-failing query 422s before
reaching here; a gate-failing registered image was never cached in the first
place, register-only `BYPASS_QUALITY_GATES` history aside):

- 25 clean false-positive pairs: `num_matches` 6..96 (max 96, 2nd-highest 84)
- 23 clean true-positive pairs: `num_matches` 96..647 (min 96, 2nd-lowest 119)

The two populations **tie exactly at 96** — one real false positive
(Animal-11), one real true positive (Animal-1's own muzzle1-vs-muzzle2) —
an irreducible ambiguity in this data, not a threshold-tuning failure. Zones
(`pipeline/lightglue_verify.py`) are set to name that tie honestly rather
than resolve it by fiat:

```
< 85       likely_different    (24/25 clean FPs; excludes the tie)
85 .. 120  ambiguous           (the tie itself: FP=96, TP=96, TP=119)
> 120      likely_same         (21/23 clean TPs; next value after the
                                 tie's neighbours jumps to 190)
```

This is 48 pairs from 9-10 real animals — revisit once more incidents
accumulate, not a large-sample calibration.

**Round 1 latency finding (superseded below, kept for the story): did NOT
meet a ~1-second budget.** Tested directly against real production crops from
the validation dataset (GPU, CUDA available, `cudnn.benchmark=False` — ruled
out as the cause since repeated calls on the identical pair stayed
consistently slow, not a one-time warmup cost):

- Small crops (~130K–550K px each side): ~400–600ms — comfortably inside budget.
- Large crops (~800K–1.7M px each side — a `crop_cattle` box that covers most
  of the frame, which a correctly-composed close-up muzzle photo routinely
  produces, since the capture UI tells officers to fill the frame): **3–5
  seconds** — well over it.

Root cause: DISK's forward pass cost scales with input pixel count, and
`crop_cattle` crops are NOT a fixed size, unlike the embedding model (always
518×518). This was flagged and deliberately NOT fixed in that pass — kept
out of the live path (`LIGHTGLUE_TIEBREAKER_ENABLED` shipped OFF) pending a
proper fix and re-validation, per this file's repeated rule about not
shipping a quality-gate/threshold change without measuring it against real
data first.

### Round 2: fixed the latency, re-validated together, now live

Two changes, made together and re-validated together — not shipped on
assumption that either one alone would be enough:

1. **Resize cap before DISK** (`pipeline/lightglue_verify.py`,
   `LIGHTGLUE_MAX_DIM`, default 512, longest side, aspect-preserving, never
   upscales). Applied inside `extract_features()`, the single place both a
   fresh (query-side) extraction and a to-be-cached (registration-side)
   extraction go through — guarantees a cached candidate's features are
   extracted exactly the way a live one would be, so there's no drift between
   the two code paths.
2. **Candidate-side feature cache** (`pipeline/muzzle_crop_cache.py`,
   `save_features`/`load_features`, alongside the existing crop cache).
   `/register` now extracts DISK features once per accepted muzzle crop
   (right after `save_crop`, same fail-open contract, skipped if the
   verifier didn't load) and caches keypoints/descriptors/image_size keyed by
   `faiss_id`. `/search`'s tiebreaker (`main._run_lightglue_tiebreaker`) tries
   `load_features()` first — on a hit, the candidate side skips DISK entirely
   and only the query is extracted live; on a miss (pre-existing registration
   before this shipped, or a past write failure) it falls back to
   `load_crop()` + live extraction on both sides, same as round 1.

**Re-validated together against the same 40-image / 57-pair dataset**, this
time running the ACTUAL production functions (`extract_features_np`,
`save_features`/`load_features`, `verify_with_cached_candidate`) rather than
a hand-rolled match function, with an explicit assertion that every pair hit
the feature cache (`cache_misses == 0`) — this benchmark is not measuring the
live-extraction fallback path. Restricted to the same "clean" population rule
as round 1 (both images pass production's quality/detection gates) for a
fair before/after comparison:

|  | round 1 (no resize, no cache) | round 2 (512px cap + feature cache) |
|---|---|---|
| clean FP `num_matches` | 25 pairs, 6..96 | 28 pairs, 28..139 |
| clean TP `num_matches` | 23 pairs, 96..647 | 23 pairs, 150..833 |
| separation | **exact tie at 96** | **clean gap, zero overlap** (max FP 139 < min TP 150) |
| latency, worst case | 3-5s | **172ms** |
| latency, overall range | 400ms-5s | 78-172ms |

Both re-validation gates passed: worst-case latency (172ms) is comfortably
under the ~1s budget, and zone separation did not regress in exchange for
speed — it *improved*, from an exact tie to a clean gap. (Plausible reason,
not proven: resizing to 512px discards some of the high-frequency texture
detail that was producing spurious keypoint matches specifically on the
false-positive pairs, while the true-positive pairs' structural agreement
survives the resize fine.) **`LIGHTGLUE_TIEBREAKER_ENABLED` now defaults to
`true`** — kept as an env var so it can still be killed fast without a
redeploy, not because it's provisional. Zone thresholds moved to **`<140`
likely_different, `140–150` ambiguous, `>150` likely_same** (see
`pipeline/lightglue_verify.py` for the full derivation) — the old 85/120/96
numbers are pre-resize and no longer apply; don't mix them with a 512px-capped
`num_matches` value.

**Still true from round 1, unchanged by round 2:** no backfill for the
existing FAISS index (only animals registered after round 2 shipped have
cached features *and* a cached crop — the fallback chips in for anyone
registered between the crop-cache and feature-cache shipping, everyone
before that gets no tiebreaker at all until re-registered), and this is 51-57
pairs from 9-10 real animals — revisit both the resize cap and the zone
thresholds once more incidents accumulate.

**Explicitly untouched, both rounds:** `/register`'s core logic (quality
gates, duplicate check), the 409 duplicate-check path, and every existing
`/search` field (`request_id`, `query_colors`, `horn_shape`, `top_matches`,
`versions`) — confirmed unchanged field-for-field via a direct Pydantic
`model_dump()` comparison, not just by inspection. `matches`/`top_matches`
construction was never touched; the tiebreaker reads `matches` after it's
finalized and only appends three new fields to the response.

## Config values worth knowing (`godhaar/config.py`)

- `IMG_SIZE=518`, `EMB_DIM=256`, `MODEL_VERSION="dinov2_arcface_v1"`.
- Encoder: DINOv2 ViT-B/14 → GeM → head(768→512→256) → L2-norm, ArcFace-trained.
  Verified discriminative weights: `best_top1.pt` (epoch 41). Deployed weights
  load from `MODEL_PATH` env (`/appstorage/Models/embedding_model/model.pt`) —
  confirm that file IS `best_top1.pt` and not a stale checkpoint.
- YOLO: `yolov8s.pt`, conf ≥ 0.30, internal conf 0.10, min bbox 5% of frame,
  max 1 cattle/image, 4-attempt retry ladder (raw → CLAHE → imgsz1280 → TTA).
- Quality: blur ≥ 20 (Laplacian var), exposure 30–225 (+std ≥ 15 for dark
  coats), min short side 96px.

## Running things locally

- venv: `.venv/Scripts/python.exe` (has torch+cuda, ultralytics, cv2, faiss).
- Encoder weights: `D:\Group Projects\Godhaar\Wildlife\for_aditya\best_top1.pt`.
- The model loads in **train mode** by default — call `model.eval()` after
  `load_checkpoint` or BatchNorm dies on batch size 1 (the server does this in
  its lifespan). To embed: `crop_cattle(img)` → `cv2.imencode(".jpg", crop)` →
  `embed_batch([jpg], model, device)`. Cosine = dot product of unit-norm 256-d
  vectors.
- To run locally outside Docker, `MODEL_PATH`/`FAISS_INDEX_PATH` env vars are
  required (lifespan raises if unset). `MODEL_PATH` must point at a real file
  (`for_aditya/best_top1.pt` works); `FaissIndex.load()` tolerates a
  non-existent `FAISS_INDEX_PATH` — just logs a warning and starts empty, so
  any placeholder path is fine for testing anything that isn't `/register`/
  `/search`. `YOLO_MODEL_PATH` is optional, defaults to a bundled name.

## CCTV counting was unreliable on real footage — full tuning history

Everything below happened on `feature/cctv-video-analytics`, on top of the
`0f2b312` CCTV commit, across several rounds. The branch is pushed — confirm
`git log --oneline -6` matches before assuming any of this is still pending.

**The original problem, on a real 48.8s field clip (herder + small herd,
never more than ~10 cattle visible in any frame):** `final_cattle_count`
(then = unique tracked IDs) read **81** — it grew almost linearly with video
duration (34 on a 15s trim, 56 on 30s, 81 on 48.8s) regardless of how many
animals were actually ever on screen at once. Root cause, confirmed
**visually** by extracting annotated frames, not just from the numbers: the
tracker was minting a new ID for the same physical animal repeatedly
(flicker), and — separately — the detector itself sometimes fires two
overlapping boxes on one animal in a single frame.

Fixes tried, each verified against the same clip before moving to the next
(diminishing but real returns — 81 → 51 unique-ID count across all of them):
1. `stable_id_iou_thresh` 0.25→0.15, `stable_id_memory_frames` 30→90
   (`PipelineConfig` dataclass defaults — these are **not** per-preset, no
   `PRESETS` entry overrides them, so this affects every preset equally).
2. New `min_frames_visible=5` config knob — drop any stable ID seen in fewer
   frames from the final count (kills pure flicker).
3. `fast` preset switched from `botsort_cattle_fast.yaml` (`with_reid:
   false`) to `botsort_cattle.yaml` (`with_reid: true`) — appearance ReID
   re-identifies cattle after occlusion. Barely moved the number (53→51) and
   barely changed processing time either — the earlier "accurate preset is
   slower" difference was mostly the higher `img_size`/no frame-skip, not
   ReID itself.
4. NMS/confidence: added `nms_iou=0.45` (new field, passed as `iou=` to
   `model.track()`/`.predict()` — wasn't wired at all before), raised
   `confidence` 0.25→0.35. Reduced the **same-frame duplicate-box** artifact
   from 3 overlapping boxes on one cow down to 2 — did not eliminate it, and
   the count plateaued around 50, not the 15-20 range a genuinely-fixed
   count should land in on this footage. **Conclusion at the time: the
   remaining inflation looks like the model itself sometimes double-detecting
   or under-segmenting close-together cattle at some angles — a real
   detection-quality question, not a threshold to keep nudging.**

**Given that, the counting methodology itself changed** rather than chasing
the detector further: `/result`'s `final_cattle_count` and `/analytics`'s
`total_cattle` were both switched to report `max_cattle_in_frame` (peak
simultaneous count — visually countable against the video) instead of the
unique-tracked-ID count, which is what all the tuning above was trying to
stabilize. `unique_tracked_cattle` stays exposed separately for whoever wants
the tracking-based figure. On the same clip this reads **9** — the tuning
work above is still real (it's what the tracker/analytics internals use to
build `unique_tracked_cattle` and the per-cow list), it's just no longer
what gets surfaced as *the* number.

**`analytics.py`'s per-cow count and `pipeline.py`'s final count can silently
disagree if you touch one without the other** — this bit twice. First
`VideoAnalytics` computed `total_cattle=len(per_cow)` completely
independently of `pipeline.py`'s flicker filter (fixed by adding the same
`min_frames_visible` filter to `_compute_per_cow`, threaded through from
`cfg.min_frames_visible` via `routes.py`'s `VideoAnalytics(...)` call). Then
the max-in-frame switch above had to touch **both** `routes.py` (`/result`)
and `analytics.py`'s `compute()` (`total_cattle=peak_count` from
`max(self._frame_counts)`) for the same reason — one file's number without
the other just moves the disagreement around instead of closing it.

**Fixed, in a follow-up commit:** `GET /history`/`GET /trends` read straight
from the `sessions` table, and `database.py`'s `save_session()` was
persisting `summary.final_cattle_count` — the pipeline-level, tracked-ID
field — while `/result`/`/analytics` had already moved to peak-in-frame.
Same job showed 9 live and 50 in `/history`/`/trends`, confirmed not
theoretical. Fix: `save_session()` now writes `summary.max_cattle_in_frame`
into the `final_cattle_count` column too (same value already going into the
`max_in_frame` column — both columns are redundant now, left as-is rather
than restructuring the schema for this). Verified end-to-end: submitted a
video, got 9/9/9/9 across `/result`, `/analytics`, `/history`, and `/trends`
for the same job.

### The annotated video was never actually playable in a browser

`pipeline.py` wrote `annotated.mp4` via `cv2.VideoWriter(fourcc=mp4v)` —
confirmed via `CAP_PROP_FOURCC` readback the file was really tagged `FMP4`
(MPEG-4 Part 2). No mainstream browser decodes that in a `<video>` tag —
confirmed directly, not just from the codec name: loaded a real output file
in Chrome, `readyState` stayed `0` forever, `canPlayType('...mp4v...')` came
back `""` while `canPlayType('...avc1...')` said `"probably"`.

The fix is not "pass a different fourcc to `cv2.VideoWriter`" — tried
`avc1`/`h264`/`H264`/`x264`, all fail identically on this machine:
```
Failed to load OpenH264 library: openh264-1.8.0-win64.dll
Could not open codec libopenh264, error: Unspecified error (-22)
```
OpenCV's bundled FFmpeg needs Cisco's OpenH264 DLL for H.264 encoding and it
won't load here — `cv2.VideoWriter.isOpened()` still reports `True` and
writes a non-empty file regardless, which is a trap: that file **also**
never leaves `readyState 0` in a real browser. Don't trust `isOpened()` or
"cv2 can read back its own file" as evidence a browser can play it — verify
in an actual `<video>` tag.

**Actual fix:** added `imageio-ffmpeg` (real, self-contained ffmpeg binary,
bundled via pip — independent of whatever codecs the host has) to
`pyproject.toml` (this repo is uv-managed, there is no `requirements.txt`).
`process_video()` now re-encodes the raw `cv2.VideoWriter` output to real
H.264 immediately after `writer.release()`:
```python
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg_path, "-y", "-i", raw_path, "-vcodec", "libx264",
                 "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
                 h264_path], check=True, capture_output=True)
os.replace(h264_path, raw_path)
```
Verified via ffmpeg's own stream probe afterward: `Video: h264 (High) (avc1
/ 0x31637661), yuv420p`. Range-request support (needed for browser
seeking/scrubbing) needed **no code change** — Starlette's `FileResponse`
(used by `GET /jobs/{id}/video`) already answers `Range` headers with a
correct `206 Partial Content` + `Content-Range`.

**One caveat that's environmental, not a code problem:** could not get a
`<video>` element to actually reach `playing` in this session's browser-
automation tooling. Before concluding the fix was broken, ran a control
experiment — generated a trivially tiny, freshly-encoded H.264 file via the
exact same `imageio-ffmpeg`/libx264 path, served it through the same
`FileResponse` mechanism, zero cross-origin variables. **It also never left
`readyState 0`.** That means this specific automated Chrome instance can't
decode H.264 at all right now, independent of anything about this fix —
every other independently-checkable signal (container structure, codec
probe via ffmpeg itself, `Content-Length` matching disk size, correct Range
handling) came back correct. Get a real visual confirmation from an actual
desktop browser before fully trusting this — the automation environment
could not provide one.

### Both counts shown side by side, not one picked as "the" answer

Reported live: a panning shot down a long goshala feeding-trough row read
`final_cattle_count: 22` when the herd visibly looked far larger than that.
The full report had a second number that wasn't being surfaced anywhere
except `/result`: `unique_tracked_cattle: 62`. Both are real, honestly
computed, and biased in *opposite* directions depending on whether the
camera is static or panning — max-in-frame (the original tuning history
above) undercounts a panning shot across a herd bigger than any single
frame ever holds; the tracked-ID count over-counts a static scene via
tracker churn (the original 81-vs-~10 problem this file already documents).
Rather than pick one as authoritative, `/analytics`, `/history`, and
`/trends` now all expose both — `total_cattle`/`final_cattle_count` (peak)
alongside `unique_tracked_cattle` (tracking) — and the goshala manager
picks whichever fits their camera setup. `AnalyticsResult.unique_tracked_cattle`
is free to compute: it's just `len(per_cow)`, already filtered by the same
`min_frames_visible` fix `pipeline.py`'s peak/tracking numbers use, so the
two stay in sync automatically rather than needing separately-maintained
logic. Session rows created before this column existed read back as `null`
for the new field via a migration (`PRAGMA table_info` + `ALTER TABLE` —
`CREATE TABLE IF NOT EXISTS` alone doesn't add a column to an existing
table) rather than erroring.

Also fixed while surfacing counts: `JobResult.output_video` and the
`sessions` DB row's `output_video` were both raw server filesystem paths
(`D:\...\annotated.mp4`) — useless to any client. Renamed to `video_url`,
set to the actual `GET /jobs/{id}/video` route. This surfaced a real latent
bug: that endpoint only ever read from the in-memory `_jobs` tracker, which
is never persisted, so a `/history` video link for any session older than
the current server process's uptime would 404 even though the DB row and
on-disk file both survive a restart. Fixed by falling back to
`db.get_session(job_id)` when the job isn't in memory. Verified by actually
downloading the video through the real endpoint (not just checking the file
exists on disk) both immediately after processing and again after clearing
`_jobs` to simulate a restart — same URL, same file, both times.

### `crowded` preset — FAST's thresholds badly undercount a packed goshala row

Reported live, on the same panning-shot clip above: even `max_cattle_in_frame`
(22) looked low against the actual video — a wide shot down a long feeding
trough visibly holds far more cattle at once than that. Confirmed by
extracting frames and counting by eye: single frames in this clip show
roughly 35-55+ cattle simultaneously, well above what `FAST` was reporting
even as its *peak*.

Root cause, confirmed empirically rather than guessed (raw-YOLO test across
3 of the densest frames, then re-verified through the full detect+track
pipeline on the whole clip — scratch scripts, not committed, results below
are what matters): `FAST`'s `confidence=0.35` (raised from 0.25 earlier in
this file's own history, but tuned against a *sparse* open-field herd to
kill duplicate-box artifacts on a single animal) was filtering out genuine,
lower-confidence detections of small/rear-view/heavily-occluded animals
further back in a packed row — **not** a false-positive problem, so
loosening it doesn't reintroduce the noise the original fix was guarding
against:

- Raw YOLO on 3 dense frames: `conf=0.35/iou=0.45` (current) found 10-13
  boxes each. `conf=0.15/iou=0.45` nearly doubled that (18-26). Adding
  `iou=0.6` pushed further (20-31).
- Full pipeline, whole clip: baseline `peak=22/tracked=62`. `conf=0.15,
  iou=0.45` reached `peak=26` but tripped ultralytics' *"NMS time limit
  exceeded"* warning on this clip — a real reliability risk, not just a
  number. `conf=0.15, iou=0.6` and `conf=0.20, iou=0.55` both reached the
  same `peak=26` without that warning; picked `0.20/0.55` as the safer of
  the two equally-good options.
- Checked for regression on a second, less-crowded clip before trusting
  this: `peak=14→16`, `tracked=20→26` — same direction, much smaller
  magnitude, no false-positive blow-up. Total detections and average
  confidence scaled proportionately with the count increase on both clips
  (not exploding independently), consistent with recovering real missed
  animals rather than adding noise.
- Tried `yolo11m` (what `BALANCED`/`ACCURATE` already use) at the same
  tuned thresholds, expecting a further gain. It didn't help —
  `peak=23`, slightly *worse* than `yolo11s`'s 26, at the same processing
  cost. Model size isn't the bottleneck here; this preset stays on
  `yolo11s`.

**Deliberately shipped as a new opt-in `Preset.CROWDED`
(`confidence=0.20, nms_iou=0.55`, otherwise identical to `FAST`), not a
change to `FAST`'s defaults.** The original `0.35/0.45` was tuned against a
genuinely different clip (the sparse open-field herd earlier in this file)
that is no longer available to re-verify against — changing the global
default risks silently undoing that fix for the case it was built for.
Same principle as showing peak vs. tracked side by side above: let the
caller pick the tool for their camera setup instead of guessing one
answer fits every scene. `nms_iou` is now also exposed as a per-request
`/analyze` override (`img_size`/`confidence`/`vid_stride` already were;
this closed the inconsistency), independent of the new preset.

**Still an honest gap, not a full fix.** `peak=26` against a ~35-55 visual
estimate on the same clip means real animals are still being missed even
after this tuning — the gains plateaued at the same `peak=26` across three
different confidence/IoU combinations, which looks like `yolo11s` approaching
its real detection ceiling on this level of shoulder-to-shoulder occlusion,
not a threshold left untuned. This generic COCO-pretrained model was never
trained on this specific scene type. If more accuracy is needed here, the
next lever is a model fine-tuned on genuinely crowded barn footage — a
data/training problem, not a config change.

### Automatic peak-vs-tracking metric selection — `final_cattle_count` picks the right one instead of always showing peak-in-frame

Follow-up to "Both counts shown side by side" above. Showing both numbers
and letting the goshala manager pick was the right call when this file
didn't yet know how to tell a static camera from a panning one — but
`/jobs/{id}/result` still had to put **one** number in
`final_cattle_count`, and it hardcoded that to `max_cattle_in_frame`
unconditionally. That's correct for a static/fixed camera (peak-in-frame
is visually verifiable, and immune to tracker-ID churn) and systematically
wrong for a panning shot, which by construction never has the whole herd
in one frame — that's the exact `22` vs `62` gap on the feeding-trough
clip above.

Preceded by a real investigation (not assumed): the two originally-suspected
causes of goshala undercounting — low-contrast dark-coated cattle/Murrah
buffalo detection failure, and wide/panning-camera coverage — were checked
against real footage before any fix was proposed. A quantitative
brightness-vs-detection-confidence correlation across ~1,350 real detections
found no dark-cattle detection problem (darkest third of detections actually
averaged slightly *higher* confidence than the lightest third — 0.610 vs
0.568 — the opposite of what a genuine low-contrast failure would look
like). The panning/counting-metric hypothesis was the one that held up,
confirmed both by the `unique_ids_so_far` growth-rate shape (see
`cctv/panning.py` below) and by directly inspecting extracted frames from
the clip in question. That result is why this section exists and the
dark-cattle path doesn't.

**Detector** (`cctv/panning.py`, `detect_panning()`): every processed frame
already logs a running `unique_ids_so_far` count to `metrics.csv` (existing
field, not new). A static camera's version of that curve rises fast while
new animals first walk into frame, then flattens once everything visible has
been seen once. A panning camera's version keeps discovering new animals
throughout, because it keeps revealing new ground. So the signal is the
*ratio* of the late-clip growth rate to the early-clip growth rate:

```
first_half_rate  = (series[n//2]      - series[0])       / (n//2)
final_qtr_rate   = (series[-1]        - series[int(.75n)]) / (n - 1 - int(.75n))
ratio            = final_qtr_rate / first_half_rate
is_panning       = ratio >= PANNING_RATIO_THRESHOLD   # 1.0
```

A ratio near or below 1 means the back quarter is growing no faster than
the first half already did (flattening — static). A ratio meaningfully
above 1 means it's still climbing at the same or a higher rate right to the
end (never flattens — panning). `PANNING_RATIO_THRESHOLD = 1.0` was picked
from real archived runs, not guessed: the confirmed-panning feeding-trough
clip (visually verified — camera visibly at a different position in frames
sampled from four different points in the clip) scored **2.19**; a
confirmed-static clip (fixed close-up on a feeding trough, same background
railing structure start to end) scored **0.51**; a third, visually-confirmed
panning clip in open pasture (camera position genuinely shifts between the
first and last sampled frame) scored **1.61**. `1.0` sits in the gap between
the static case and both panning cases with real margin on each side, not
splitting the difference between two close numbers. Two guard conditions
(`MIN_FRAMES_FOR_DETECTION = 20`, `MIN_UNIQUE_IDS_FOR_DETECTION = 4`) default
to "not panning" on a clip too short or too sparse to say anything — an
unclear signal should never flip the metric away from the visually-checkable
default.

**Wiring** (`cctv/pipeline.py`, `cctv/routes.py`): `VideoSummary` gained
`is_panning`, `panning_ratio`, `count_method_used` as new fields, computed
alongside (not replacing) the existing `final_cattle_count`/`count_method`
pair, which keep their pre-existing, unrelated meaning (`cfg.use_tracking`
driven) — nothing about the old computation changed. The actual selection
happens only where the API response is built, in `/jobs/{id}/result`:

```python
final_cattle_count=(
    s.unique_tracked_cattle if s.is_panning else s.max_cattle_in_frame
),
```

`unique_tracked_cattle` was already the right panning-case number and
already had its own flicker filter (`min_frames_visible`, from the tracking
fix earlier in this file) — this doesn't add a second counting method, it
just decides which of the two already-computed, already-validated numbers
to report as *the* number. `count_method_used` (`"peak_in_frame"` |
`"tracking_estimate"`) rides alongside so which one fired is inspectable
without re-deriving it from `panning_ratio` — additive, diagnostic, not
surfaced in the dashboard UI. `max_cattle_in_frame`/`unique_tracked_cattle`
are both still returned too, so nothing that reads the old side-by-side
fields breaks.

**This is entirely additive and entirely inside `inference_server`.**
`JobResult`/`SessionInfo` gained fields, nothing was removed or renamed;
`sessions` got 3 new nullable columns via the same `PRAGMA table_info` +
`ALTER TABLE` migration pattern used for the peak/tracking columns above, so
old rows read back as `NULL`/`None` rather than erroring.
`godhaar_analytics_dashboard` and `go-apiserver` need zero changes for this
to be correct — go-apiserver's Go client reads `max_cattle_in_frame` by
field name today and never touches `final_cattle_count` at all, so this
fix's correctness doesn't yet reach the dashboard until that client is
updated to prefer `final_cattle_count`/`count_method_used`; flagged, not
fixed, since touching go-apiserver was out of scope for this change.

Unit tests in `cctv/tests/test_panning.py` pin the three real-data ratios
above as regression cases (via a helper that reconstructs just the 4 index
values `detect_panning()` actually reads — `cctv/runs/` is gitignored, so
the fixtures don't depend on those files existing) plus synthetic edge
cases (too-short clip, too-few-IDs, exact-threshold boundary, flat-then-late
growth). Full real-clip validation (all 23 clips in `cctv/clips/`, plus a
`CROWDED_HD` vs `FAST` cost/latency comparison) — see the note directly
below.

### Still open: does low inference resolution look like panning on a wide static shot?

One real clip in the 23-clip validation batch (`clip_01.mp4`) — visually
confirmed static (same background structure across sampled frames, no
camera motion) during the dark-cattle/panning investigation above — came
back `is_panning=True` (ratio 1.234) when run through the actual
`detect_panning()` pipeline at `FAST`'s 640px inference size. Not yet
resolved at the time of writing: the working hypothesis is that small/
far-from-camera animals in a wide static shot get inconsistent
frame-to-frame detections at low resolution (flicker in, flicker out),
which mints new stable IDs steadily throughout the clip and mimics the same
"never flattens" curve shape genuine panning produces — a resolution
confound, not a wrong classification of real camera motion. `clip_01` is
also in the `CROWDED_HD` (1920px) comparison subset for the same batch, so
this will be checked directly: if `clip_01` classifies as static at
`CROWDED_HD` and panning only at `FAST`, that confirms the resolution
hypothesis and is worth knowing before trusting `is_panning` blindly on
low-res wide shots. If it's still `True` at `CROWDED_HD` too, the clip
likely has some real camera adjustment that wasn't visible in the sampled
frames, and the classification is probably correct after all.

### Two more correctness bugs found after the tuning history above — same-frame duplicate IDs, and static-position occlusion matching

Found 2026-08-10, independently (validated first against a standalone
copy of this same tracker, before discovering `feature/cctv-video-analytics`
had already merged here — see "an abandoned parallel path" below), while
chasing the same `unique_tracked_cattle` overcounting problem the tuning
history above already spent real effort on. **These are not more
threshold nudging on the axis that plateaued at ~50-51 above** — they're
two actual logic bugs in `stable_id.py`'s matching, found by looking at
*which* IDs were short-lived and clustered in tight frame windows, not by
sweeping more numbers:

1. **`StableIdMapper` could hand the same stable ID to two different
   boxes in the same frame.** The direct `raw_id → stable_id` cache
   (`_resolve`'s step 1) was trusted unconditionally, even when a
   *different* raw ID had already claimed that same stable ID earlier in
   the very same frame — which happens once BoT-SORT reassigns a fresh
   raw ID mid-track and the old raw ID later reappears (exactly the
   flicker this file's tuning history already describes). Verified on a
   real 15s clip (9 cattle max in frame): **37 of 225 processed frames
   had a stable ID stamped on two live boxes at once.** That's not a
   counting error, it's misidentification — whichever ID got
   double-booked has its speed/trajectory/isolation analytics silently
   corrupted, jumping between two different animals' positions. Fixed
   with a `claimed_this_frame` guard in `update()`: a stable ID already
   taken this frame is never reused, a new one is minted instead. **0
   duplicate-ID frames after the fix,** re-confirmed on every run since.

2. **Matching compared a returning cow's box against its last known
   STATIC position**, with no allowance for the animal having walked
   during the gap. `stable_id_memory_frames=90` (already correctly set
   in this branch's `config.py`, independently aligned to the tracker's
   own `track_buffer`) makes the memory window long enough — but a cow
   walking steadily for even ~1s moves far enough that IoU against its
   *old* position drops under `stable_id_iou_thresh` regardless of how
   long the memory lasts. Fixed by projecting each remembered box
   forward using that ID's own last observed velocity before computing
   IoU (`_project` in `stable_id.py`), rather than comparing against a
   frozen coordinate. Verified with a synthetic case first (a cow
   walking at constant velocity, occluded 42 raw frames, handed a new
   raw ID — correctly resolves back to the same stable ID; confirmed it
   would NOT have under the old static-position logic), then on real
   footage: **`unique_tracked_cattle` on the same test clip went from 41
   (after fix #1 alone) to 33**, `max_cattle_in_frame` unchanged at 9.

**Neither fix changes what's shown as the primary number**
(`total_animals`/peak-in-frame, per "both counts shown side by side"
above) — they improve `unique_tracked_cattle`/`total_clear_animals`, the
secondary figure. 33 is still not validated against a labeled
ground-truth count — same caveat as everywhere else tracking numbers
appear in this file.

### `CATTLE_TERMS` only ever matched "cow"/"horse" — silently missing water buffalo

Checked against the actual `yolo11s.pt` COCO class list: `"cattle"`,
`"bull"`, `"calf"`, `"buffalo"`, `"bovine"`, `"ox"` match **nothing** —
COCO (80 classes) has no such names, so those six entries in
`CATTLE_TERMS` were dead vocabulary that never matched a real detection.
Only `"cow"` (19) and `"horse"` (17) ever fired. That silently
undercounts water buffalo, extremely common in Indian goshalas, since
COCO has no buffalo class and a generic detector routinely misclassifies
one as something else instead.

**This exact problem was already found and fixed once in this repo** —
`pipeline/yolo_crop.py` (the OTHER service, muzzle detection) matches
classes `{17,18,19,20,21}` = horse/sheep/cow/elephant/bear specifically
"to catch buffalo misclassifications" (see this file's earlier section on
that fix). Applied the identical fix here — `CATTLE_TERMS` now also
includes `sheep`/`elephant`/`bear` — rather than re-deriving it blind.
Verified no regression on the real test clip (identical
`max_cattle_in_frame=9`, `total_detections=1410`,
`average_confidence=0.7328` before/after, since that clip has no
buffalo/misclassified animals to trigger the difference) — this fix can't
be proven to *help* without footage that actually contains buffalo, only
proven not to hurt cow-only footage.

### An abandoned parallel path, for context if it resurfaces

Before discovering `feature/cctv-video-analytics` was already merged,
this session built a **separate, standalone** copy of the same
cattle-counting app (`cctv_service/`, copied from a now-stale
`D:\Group Projects\cattle_ai_project`) as its own FastAPI process,
including the same two `stable_id.py` fixes above, independently. That
whole path was abandoned once the merge was found — the fixes were
ported into the real, already-merged `cctv/` package instead (this
section), and `cctv_service/` was deleted. If old references to
`cctv_service/` or `cattle_ai_project` turn up anywhere (docs, chat
history, stray local folders), they're describing this dead end, not the
real running service.

## `CROWDED_HD` is opt-in per camera, not a safe global default — tested, not assumed

2026-08-17: `CROWDED_HD` (1920px) had briefly become the system-wide
default (both `make_config()`'s default arg and `/analyze`'s `preset`
Form default) as part of the decoupled classify/count pipeline change
above. That was wrong, and reverted — back to `Preset.FAST` as the
default, with `CROWDED_HD` now only reachable per-camera via
`cctv/config.py`'s `LOCATION_PRESET_OVERRIDES` map (keyed by
`location_tag`) or an explicit `preset` override on a single `/analyze`
call.

**Why, with real evidence, not intuition** — 2 real goshala clips run
head-to-head at `FAST` vs `CROWDED_HD`, same clips, same day:

- **Dense night feeding-trough scene: genuine improvement.** Peak count
  17 → 43. Confirmed visually (annotated frame at the same timestamp,
  both resolutions) that `CROWDED_HD` recovers real animals `FAST`
  misses outright in the crowded cluster — not just more boxes, actual
  previously-undetected cattle. Confidence also rose slightly (0.53 →
  0.57). Cost: 2.1x processing time (46s → 97s for a 48s clip).
- **Clean, low-density daytime corridor scene: regressed.** Peak count
  held at 2 either way, but that's misleading on its own — average
  detection confidence DROPPED 0.86 → 0.74, and total detections rose
  30% (600 → 782) for the identical answer. `CROWDED_HD`'s lower
  confidence threshold (0.20 vs `FAST`'s implicit higher operating
  point) and looser NMS (0.55) generate more, noisier, lower-confidence
  boxes even in a scene with nothing to gain from the extra resolution
  — the peak count only survived because those extra boxes never
  coincided in a single frame. A busier or slightly-worse-lit version of
  the same clean scene could easily have inflated the peak count on
  nothing.

Same pattern the panning-detection section above already found for a
different symptom (`CROWDED_HD`'s confidence/NMS corrupting the panning
signal) — this is the general case: `CROWDED_HD`'s settings are tuned
for density, and applying them to a scene that isn't dense doesn't just
"waste compute," it measurably degrades detection quality. **Don't
re-promote `CROWDED_HD` to a global default on the strength of the
crowded-scene win alone — the clean-scene loss is just as real,
measured the same way, same day, same method.**

`cctv/config.py::resolve_preset(location_tag, explicit_preset)` is the
one place this decision gets made: explicit `preset` always wins (manual
re-runs, testing); otherwise `location_tag` is looked up in
`LOCATION_PRESET_OVERRIDES`; anything not flagged there falls through to
`DEFAULT_PRESET` (`FAST`), never to `CROWDED_HD`. The map ships empty —
no location_tag is flagged yet, since no real camera/location naming
convention was available to populate it correctly rather than guess. Add
entries there once specific crowded feeds (feeding troughs, entry gates
during herding) are identified by `location_tag`.

### Ground truth at peak density, and why NMS tuning turned out not to be the fix

The FAST-vs-CROWDED_HD numbers above (17 vs 43 peak-in-frame) were never
checked against real ground truth — just against each other. Worth doing
before trusting either: manually counted a raw, unannotated frame from
clip 2 (frame 704, the source frame that actually produced CROWDED_HD's
peak of 43 — NOT the same frame FAST peaked on; FAST's own peak of 17
happens at frame 222, ~19s earlier in the clip, a genuinely different
moment). Counted twice independently, region by region, specifically
checking for occlusion/cut-off animals: **21 and 25**, converging on
**~21-25 (midpoint ~23)**. At that same frame 704, FAST itself reads only
13 (not its own clip-wide peak of 17 — this is FAST's reading at the
literal same moment CROWDED_HD peaked).

**The correct reading of these three numbers is NOT "CROWDED_HD (43) is
closer to ground truth than FAST (13)."** It isn't — CROWDED_HD's error
(~87% over ~23) is roughly DOUBLE FAST's error (~43% under ~23). An
earlier pass at this got that comparison wrong by treating "same side of
a threshold" as "better," which doesn't hold once you look at the actual
percentages. The real reason to start any fix from CROWDED_HD rather than
FAST is structural, not "closer": NMS and confidence filtering can only
REMOVE boxes a detector already proposed. FAST's lower-resolution pass
never proposes the boxes needed to close its 43%-under gap — there's
nothing to tune upward. CROWDED_HD's 87%-over gap, by contrast, is at
least the right kind of error to attack with post-processing, since the
boxes already exist and the question is which of them are real.

**Tested that hypothesis directly — swept `nms_iou` from the CROWDED_HD
default (0.55) down to 0.35, all else held at CROWDED_HD's settings,
same frame 704 each time:**

| nms_iou | count @ frame 704 | vs. ground truth (~23) |
|---|---|---|
| 0.55 (baseline) | 43 | +87% |
| 0.50 | 42 | +83% |
| 0.45 | 42 | +83% |
| 0.40 | 41 | +78% |
| 0.35 | 41 | +78% |

**Result: NMS tightening is not the fix.** The sweep closed only 2 of the
~20-count gap and flattened out by 0.40 — going tighter did nothing
further. Checked for the failure mode this could plausibly cause instead
(genuinely adjacent, distinct animals getting merged into one box at the
tighter setting) — not observed; isolated animals stayed isolated at
0.35, same as baseline. So tightening this far is *safe*, just
ineffective. Also re-ran the clean daytime clip (clip 3) at
`nms_iou=0.35`: results came back byte-identical to its 0.55 baseline (2
peak, 4 tracked, 0.7418 confidence, 782 detections) — no regression, but
also because that clip has too few overlapping detections for NMS
threshold to matter either way.

**Why NMS specifically doesn't help**: IoU-based NMS only suppresses a
box that geometrically overlaps enough with another surviving box.
Barely moving the count when swept this hard suggests the ~20-count gap
at peak density isn't classic duplicate-box-on-one-animal (which NMS
targets) — it's more likely spatially distinct, low-confidence phantom
detections in the dark, dense cluster (background/shadow/hay misread as
cattle) that never overlap enough with a real box to be NMS's problem at
all. That points at `confidence` (currently 0.20, quite permissive) as
the more plausible next lever — **not yet tested, a hypothesis only,
flagged here so it isn't re-derived from scratch, not a recommendation**.

**Not shipped as a result of this investigation: no `nms_iou` default
change.** The evidence doesn't support one. If someone reads the
box-stacking symptom in a `CROWDED_HD` frame and reaches for NMS as the
obvious fix, this section is why that specific fix was already tried and
didn't work.

### `confidence` threshold: much more effective on the crowded case, but fails the clean-case check outright

The flagged next hypothesis above (raise `confidence` from `CROWDED_HD`'s
0.20, since NMS couldn't touch the gap) — tested the same way, same
frame 704, `nms_iou` held at CROWDED_HD's own default (0.55) throughout:

| confidence | count @ frame 704 | vs. ground truth (~23) |
|---|---|---|
| 0.20 (baseline) | 43 | +87% |
| 0.25 | 39 | +70% |
| 0.30 | 34 | +48% |
| 0.35 | 31 | +35% |
| 0.40 | 27 | +17-29% |

Far more effective than NMS — 0.40 gets close to ground truth, closing
most of the gap NMS couldn't move at all.

**Visual safety check at 0.40** (comparing the same frame, baseline vs.
0.40, by spatial position rather than trusting tracker IDs, which aren't
stable across separate runs): the clear, confident animals (isolated
walking cow 65%, isolated resting cow 82%, the bottom-left pair 88%/83%)
were completely untouched. What disappeared was concentrated almost
entirely in the 22-39% confidence band, in the darkest/most ambiguous
part of the cluster. One borderline 26% detection at the frame edge
mapped to the exact spot where this investigation's own two independent
ground-truth counting passes had disagreed — the model's uncertainty
matched a real human's uncertainty there, not a confident miss. On this
check alone, 0.40 looked safe.

**Then the clip 3 (clean daytime) regression check failed decisively —
not marginally.** Checked frame-by-frame, not spot-checked:

| | FAST | CROWDED_HD (0.20) | CROWDED_HD (0.40) |
|---|---|---|---|
| Peak-in-frame | 2 | 2 | **1** |
| Frames with 2+ cattle detected (of 575) | - | **207 (36%)** | **0 (0%)** |

At `confidence=0.40`, the model never detects both real cattle together
in a single frame anywhere in the entire clip — not an edge case, the
whole clip. One of the two animals (visually confirmed real earlier in
this investigation) is systematically suppressed throughout a
well-lit, low-density scene, the exact failure mode this check existed
to catch: a real animal whose confidence is genuinely, persistently
below the raised floor.

**Not shipping any `confidence` default change either.** Same
disqualifying criterion as the NMS result: a setting that fixes the
crowded case by breaking the clean case outright isn't a fix. Had this
passed both checks it would have become another parameter on the same
`LOCATION_PRESET_OVERRIDES` per-camera system already built (`cctv/
config.py`) — not a new mechanism — but it didn't pass, so nothing was
wired in. `LOCATION_PRESET_OVERRIDES` still only selects a `Preset`, not
individual `confidence`/`nms_iou` values, and stays that way until a
setting actually clears both checks.

**Where this leaves the ~20-count peak-density gap**: neither of the two
cheap, obvious post-processing levers (NMS, confidence) closes it
without a corresponding real cost elsewhere.

### Closed, for now: parameter tuning is done on this, mitigation is camera placement

Deliberate decision, not an oversight: **not chasing the untested
0.25-0.40 confidence gap noted above, or any further NMS/confidence
sweep.** Both parameters this crowded-cluster investigation had reason
to try are now tested, with real ground truth and real safety checks,
not guessed at:

- **`nms_iou`**: safe (no adjacent-animal merging, no clip 3 regression)
  but ineffective (closed 2 of ~20 count gap). Not worth further sweeping
  — the 0.55->0.35 range already covers the plausible space and it
  flattened out by 0.40.
- **`confidence`**: effective on the crowded case (43->27, close to
  ground truth) but fails the clean-case check outright (0/575 vs
  207/575 frames detecting both real animals in clip 3) — a real animal
  suppressed for an entire clip, not a marginal cost. Disqualified, not
  marginal.

Both were tested at CROWDED_HD's resolution — the actual ~20-count gap
at extreme peak density (43 vs ~23 ground truth) does not have a cheap
post-processing fix in this pipeline as it exists today. That's a real
finding, not a gap in the search: two different mechanisms (box overlap
suppression, confidence filtering) were tried, one failed to help, the
other helped by breaking something else — consistent with the remaining
error being neither "duplicate boxes on one animal" (NMS's job) nor
"real animals scored too low" alone (confidence's job), but genuine
model uncertainty at extreme density that a threshold can't sort
correctly either direction.

**The practical mitigation is camera placement, not more parameter
search**: multiple cameras giving narrower coverage of a dense cluster
(e.g. two angles on a crowded feeding trough instead of one wide shot
trying to cover the whole thing) reduces how many animals any single
camera ever has to resolve at once, which is the actual constraint this
investigation ran into — not a tunable one. `LOCATION_PRESET_OVERRIDES`
(`cctv/config.py`) is still the right mechanism for the resolution
question (CROWDED_HD vs FAST per camera) — this doesn't change that,
it just closes out NMS/confidence as dead ends for anything beyond
what's already shipped.

## The real leave-one-out recall test — 218 real animals, genuinely held-out query photo

Every recall/precision number in this file up to this point came from either
a small WhatsApp sample or was inferred rather than measured end-to-end
against a photo the system had never seen. This is the first real
leave-one-out test: register each of the 218 real Uttarakhand animals, then
search using a muzzle photo (`muzzle3.jpg`) that was **never embedded during
registration** — a genuinely unseen query, not a re-submission of a
registered photo. `LIGHTGLUE_TIEBREAKER_ENABLED=true` for the whole run, so
this measures the actual current system, not the pre-LightGlue baseline.
Script: `uk_leave_one_out.py`; raw logs (`register_log.json`,
`search_log.json`, 161 entries each) kept in the scratchpad, not committed.

**Unavoidable compromise, stated up front:** `/register` hard-requires
exactly 3 muzzle images (`len(muzzle_images) != 3` → 422) — there is no way
to submit 2. So the 3rd slot was filled with a **duplicate of muzzle1**
(`muzzle1, muzzle2, muzzle1`), never muzzle3 — the held-out photo's bytes are
never sent anywhere during registration, which is the actual requirement.
Two downstream effects of this, both real and both accounted for below
rather than hidden:
- The muzzle-color majority-vote gate sees muzzle1 twice, so muzzle1's own
  color reading auto-wins any 2-of-3 vote regardless of muzzle2 — likely
  passes MORE animals through registration's color-consistency check than
  true independent 3-photo registration would. (In practice this run's 30
  `body_color_inconsistent` rejections were all **front1/front2**
  disagreements, a separate gate the duplication doesn't touch — see below.)
- Two embeddings per animal are byte-identical (muzzle1 = muzzle1), so any
  search that ranks back into its own animal has a guaranteed rank1/rank2 (or
  rank2/rank3) tie at identical score, artificially shrinking the measured
  gap between an animal's own top match and its next-best. Confirmed directly
  in a smoke test before the full run: two identical scores
  (0.8549675941467285) at gap=0.0. This likely suppresses the MATCH-vs-REVIEW
  split (MATCH requires `gap >= 0.08`) below what true independent triplicate
  photos would produce — so the 12/161 strict-MATCH count below is plausibly
  a **lower bound** on what independently-photographed registration would
  achieve, not an exact prediction of it.

**Decision replication, also stated up front:** `inference_server` itself
only returns raw `top_matches` (faiss_id/score/rank/gap) — MATCH/REVIEW/
UNKNOWN is decided in go-apiserver's `decision.go`, a separate repo. The
test script replicates it exactly for score+gap
(`matchThreshold=0.82, reviewThreshold=0.72, gapThreshold=0.08`) and for the
LightGlue demotion (`applyLightglueDisagreement`: MATCH→REVIEW or REVIEW→
UNKNOWN when `lightglue_zone == "likely_different"`, never a promotion).
**Deliberately NOT replicated:** `decision.go`'s color/horn attribute
nudge (`attributeWeight=0.03`) — that's a separate, already-tested piece of
the decision, and this test is scoped to raw embedding recall and
LightGlue's effect specifically. `lightglue_checked`/`lightglue_zone` are
read directly from the real server response for each search, not simulated.

### Registration: 161/218 succeeded (73.9%), all 57 failures are real 422 quality gates

No forced/assumed pass rate — this is what the real 218-photo set actually
did against the real registration pipeline, cold:

| Failure reason | Count |
|---|---|
| `body_color_inconsistent` (front1 vs front2 disagree) | 30 |
| `RECAPTURE_NO_DETECTION` (muzzle not found) | 10 |
| `RECAPTURE_MULTI_CATTLE` (more than one animal in a muzzle frame) | 9 |
| `bad_quality` (blur) | 8 |
| **Total failures** | **57** |

All 57 are genuine `422`s from real gates — nothing failed for an
infrastructure reason. `body_color_inconsistent` alone is over half of all
failures and is **not** an artifact of the muzzle-duplication workaround
(see above) — it's a front1/front2 color-classifier disagreement, a
pre-existing gate this test didn't touch. The 161 successful registrations
are what the leave-one-out search below runs against.

### Search: 161 leave-one-out queries against the held-out muzzle3.jpg

Every one of the 161 registered animals was searched with its own held-out
`muzzle3.jpg` + `front1.jpg`, against the full 161-animal candidate pool.
Bucketed by whether the top-1 match was the animal's own record and what the
replicated decision (score+gap, then LightGlue demotion) landed on:

| Bucket | Count | Meaning |
|---|---|---|
| `TRUE_MATCH` | 12 | Correct animal, auto-confirmed MATCH |
| `TRUE_REVIEW` | 85 | Correct animal, surfaced as REVIEW for an officer to confirm |
| `CORRECT_ANIMAL_BUT_UNKNOWN` | 5 | Correct animal was top-1, but decision fell to UNKNOWN — lost, would read as "not registered" |
| `WRONG_ANIMAL` | 47 | Top-1 was a different animal (any decision level) |
| `REJECTED_PRE_MATCH` | 12 | Query itself failed a quality/detection gate before any match was attempted |
| `NO_CANDIDATES_RETURNED` | 0 | — |
| **Total** | **161** | |

**Recall — two honest numbers, not one rounded story:**
- **Strict (auto-MATCH only): 12/161 = 7.5%.** This is the fraction where the
  system would confirm a match with zero human involvement.
- **Robust (correct animal surfaced at all, MATCH or REVIEW): 97/161 = 60.2%.**
  This is the fraction where an officer reviewing the REVIEW queue would see
  the *correct* animal as the candidate to confirm — the number that matters
  if REVIEW is actually staffed and used as intended, not rubber-stamped.
- The gap between these two is exactly the muzzle-duplication artifact's
  predicted effect (suppressed gap → most correct top-1 matches land as
  REVIEW, not MATCH) — expected given the compromise above, not a surprise
  finding.

### Precision: LightGlue's actual effect on the 47 wrong-animal cases, measured, not assumed

Before LightGlue's demotion is applied, the raw score+gap classifier alone
already put most wrong-animal cases at REVIEW, not MATCH — go-apiserver's
threshold design already does a lot of the work:

| | Count |
|---|---|
| Wrong-animal cases classified REVIEW by score+gap alone (before LightGlue) | 44 |
| Wrong-animal cases already UNKNOWN by score+gap alone (LightGlue irrelevant) | 3 |
| Wrong-animal cases where LightGlue demoted REVIEW→UNKNOWN | 39 |
| Wrong-animal cases LightGlue did **not** catch — still REVIEW after demotion | 5 |

So of 47 wrong-animal cases: **42 end up UNKNOWN (never shown to an officer
as a candidate at all)**, and **5 still reach REVIEW** — meaning an officer
would see a wrong candidate offered for manual confirm/reject in those 5
cases. All 5 are listed here for the record (`lg_zone` is what LightGlue
itself measured — 4 came back `likely_same`, i.e. LightGlue was actively
wrong, not just cautious; 1 was `ambiguous`):

| Query animal | Wrongly matched to | `lightglue_zone` |
|---|---|---|
| UKDEGR827871 | UKDEGR455918 | likely_same |
| UKDEJS014531 | UKDEJS424252 | likely_same |
| UKDEJS155191 | UKDEJS240379 | ambiguous |
| UKDEJS698982 | UKDEJS219566 | likely_same |
| UKDEJS987250 | UKDEJS250469 | likely_same |

**LightGlue does cost true positives, measured exactly: 4/102 correct-animal
cases (3.9%)** were demoted from REVIEW to UNKNOWN because LightGlue called
`likely_different` on a photo pair that was, in fact, the same animal:

| Query animal | score | gap |
|---|---|---|
| UKDEJS280448 | 0.8081 | 0.0000 |
| UKDEJS725951 | 0.8718 | 0.0272 |
| UKDEOT336574 | 0.7834 | 0.0236 |
| UKDESH010833 | 0.8246 | 0.0021 |

These 4 are exactly what the `CORRECT_ANIMAL_BUT_UNKNOWN` bucket (5) is made
of, minus 1 case that was already UNKNOWN before LightGlue ran regardless
(so LightGlue's net cost to recall in this run is these 4, not all 5).

**Precision, at the level an officer actually experiences it:**
- **At auto-MATCH: 12/12 = 100% in this run** — no wrong-animal case survived
  score+gap+LightGlue all the way to auto-MATCH. Small N (12), so this is
  encouraging, not proof of a hard ceiling.
- **At REVIEW: 85 correct / 90 total REVIEW (85 true + 5 wrong) = 94.4%.** An
  officer working the REVIEW queue sees the right animal roughly 19 times out
  of 20; the other 1 in 20 needs the officer's own judgment to reject
  (both visually-checked highest-confidence wrong-match examples in an
  earlier pass this session were confirmed genuinely different animals by
  eye, not dataset duplicates — this system is not just tie-breaking on
  near-identical photos of the same animal under a different code).

### Combined picture

Of 161 genuinely-unseen queries: **97 correctly point an officer at the
right animal (12 hands-free, 85 needing a human nod), 47 point at the wrong
animal (42 of those silently filtered to UNKNOWN before anyone sees them, 5
surfaced and needing a human to catch the error), 12 never got a match
attempt at all (query itself failed a quality gate), and 5 correct matches
were lost to UNKNOWN (4 by LightGlue's own false call, 1 by score+gap
alone).**

This supersedes the earlier "2/5 false positives" precision figure quoted
elsewhere in this session's history — that number was measured **before**
LightGlue's demotion existed in go-apiserver. Both of those original false
positives had `lightglue_zone=likely_different` when re-checked, so the
current full system's effective precision on that exact sample is 0/5, not
2/5. Raw-embedding precision and full-current-system precision are different
numbers; always state which one is being reported.

**What this does and doesn't prove:** this is real data run through the real
pipeline end-to-end with a genuinely held-out photo — not a projection. The
one caveat that should travel with every number above is the
muzzle-duplication compromise: registration here used 2 independent photos +
1 duplicate, not 3 independent photos, which plausibly understates the true
MATCH rate (gap suppression) while not obviously biasing the WRONG_ANIMAL
rate either direction. If a true 3-independent-photo comparison is ever
needed, it requires either a 4th real photo per animal or a change to
`/register`'s hard 3-image requirement — neither was in scope here.

## Correction: the leave-one-out test above under-measured recall — the FULL production stack gives a materially better number

The test above deliberately excluded `decision.go`'s attribute (color/horn)
term. Replicating it and re-bucketing the same 161 leave-one-out searches
found something more important than the attribute term's own effect: **the
original test's score+gap replica itself didn't match production**, and
correcting that alone — before the attribute term does anything — moves
strict recall from 12/161 (7.5%) to 40/161 (24.8%).

**Root cause of the discrepancy:** the original replica computed each
search's gap from `inference_server`'s raw response field
(`top_matches[0].gap`), which is a **per-embedding** pairwise gap —
rank-1-embedding-score minus rank-2-embedding-score, with no notion of which
embeddings belong to the same animal. `decision.go`'s real
`decideOnRawScores()` never does this: it first aggregates every returned
embedding to **per-animal max score** (`cattleScores[gid] = max(...)`, in
`routes.go`), and only computes the gap between the top two *animals* after
that aggregation. The leave-one-out registration compromise (muzzle1
duplicated into the 3rd slot — see above) means every animal has two
byte-identical embeddings, which are near-guaranteed to land at rank 1 and
rank 2 of the SAME animal's own results. The original replica measured the
gap between those two near-duplicate embeddings — near zero, by
construction — and called it the animal's confidence gap. Production never
does this: aggregating to one max score per animal *before* computing the
gap means a duplicate embedding of the correct animal can never suppress its
own gap, because it collapses into the same single entry its sibling
embedding already occupies. The muzzle1-duplication compromise is still real
and still limits what this test can prove about true 3-independent-photo
registration (see the caveats above, unchanged) — but the specific "gap
suppression tanks strict recall" effect reported earlier was **mostly a bug
in the test's decision replica, not a property of the real system.**
Confirmed directly: cases like `UKDEHF533712` had `old_gap=0.0` under the
flawed per-embedding replica and `new_gap=0.249` under correct per-animal
aggregation — nowhere close to the same number.

Lesson, worth generalizing: **when replicating a decision engine for a test,
replicate its literal aggregation order, not just its final threshold
constants.** Getting `matchThreshold`/`gapThreshold` numerically right was
not enough — the *shape* of what those thresholds are applied to (embedding
scores vs. per-animal max scores) changed the answer by more than the
attribute/LightGlue layers combined.

### The full stack, correctly replicated: score+gap (animal-aggregated) → attribute nudge → LightGlue demotion

`full_stack_replay.py` (scratchpad) re-derives everything offline from the
already-captured `register_log.json`/`search_log.json` — no new API calls —
using `go-apiserver`'s exact `rankCandidates`/`decide`/
`applyLightglueDisagreement` (`decision.go`): per-animal max-score
aggregation, `attributeWeight=0.03` color/horn/muzzle-color agreement
(`[-1,1]`, only over attributes present on both sides), the real
adjusted-vs-raw "take the less confident" merge (and the reported animal is
always the **attribute-adjusted** top candidate, even when the *decision
level* falls back to the raw one), then LightGlue's demote-only step on top.
`searchTopK=5` (routes.go) is production's own limit, not a shortcut this
test introduced — the captured `top_matches` already match what
go-apiserver itself receives.

**Full-stack bucket breakdown (161 searches, same data as before):**

| Bucket | Count |
|---|---|
| `TRUE_MATCH` | 38 |
| `TRUE_REVIEW` | 61 |
| `CORRECT_ANIMAL_BUT_UNKNOWN` | 9 |
| `WRONG_ANIMAL` | 41 |
| `REJECTED_PRE_MATCH` | 12 |

**Recall, the real complete-system numbers:**
- **Strict (auto-MATCH): 38/161 = 23.6%** — up from the partial replica's
  7.5%, mostly the aggregation fix, not the attribute term.
- **Robust (MATCH+REVIEW correct): 99/161 = 61.5%** — close to the earlier
  60.2%, because the earlier number's REVIEW-inclusive recall was already
  fairly forgiving of the gap-suppression bug; the fix mainly moves cases
  from REVIEW to MATCH, not from UNKNOWN to REVIEW.

**Precision, the real complete-system numbers:**
- **At auto-MATCH: 38/38 = 100%.** No wrong-animal case reached MATCH
  anywhere in this run, at any stage of the stack.
- **At REVIEW: 61/64 = 95.3%** (3 wrong-animal cases still reach REVIEW even
  after the full stack — see below).
- **Combined, everything an officer is ever shown (MATCH+REVIEW): 99/102 =
  97.1%.**

### Isolating each layer's real contribution — not just the combined number

Re-bucketing the SAME 161 searches at each stage (raw score+gap only, then
+attribute, then +LightGlue = the table above) separates what each signal
actually does, rather than reporting only the end state:

| Stage | MATCH | REVIEW | UNKNOWN (correct) | WRONG | REJECTED |
|---|---|---|---|---|---|
| A: raw score+gap (animal-aggregated) | 40 | 61 | 1 | 47 | 12 |
| B: + attribute nudge | 38 | 68 | 2 | 41 | 12 |
| C: + LightGlue (full stack) | 38 | 61 | 9 | 41 | 12 |

**Attribute layer (A→B) — net positive, small and bounded cost:**
- **Fixed 6 wrong-top-1 cases to the correct animal** by reordering on full
  color+muzzle+horn agreement (`agreement=1.0` in every case:
  `UKDEJS155191`, `UKDEJS211897`, `UKDEJS698982`, `UKDEJS791332`,
  `UKDEOT569293`, `UKDESH004551`) — this is the direct explanation for
  `WRONG_ANIMAL` dropping 47→41.
- **Cost: 2 true positives demoted MATCH→REVIEW** (`UKDEGR942545`,
  `UKDEJS137483`) — never lost, still correctly flagged for a human to
  confirm, just less confidently.
- **Cost: 0 identity losses, 0 additional UNKNOWNs.** The stage-A→B
  UNKNOWN count going 1→2 is not a new loss — it's `UKDEOT569293` being one
  of the 6 reordering fixes above (its identity became correct, but its
  decision level was already going to be UNKNOWN regardless).
- Net: attribute agreement only ever helped or was neutral for true
  positives in this run; its only real cost is confidence-level, never
  identity.

**LightGlue layer (B→C) — the one signal that costs real true positives:**
- **Caught (further suppressed) 35 of the 41 wrong-animal cases** that
  survived score+gap+attribute.
- **3 wrong-animal cases were already UNKNOWN before LightGlue ran** —
  score+attribute alone was enough, LightGlue moot for these.
- **3 wrong-animal cases are NOT caught by anything in the full stack** —
  these still reach REVIEW and would need a human to reject them:

  | Query animal | Wrongly matched to | `lightglue_zone` | agreement | score |
  |---|---|---|---|---|
  | UKDEGR827871 | UKDEGR455918 | likely_same | -0.33 | 0.771 |
  | UKDEJS014531 | UKDEJS424252 | likely_same | 1.0 | 0.855 |
  | UKDEJS987250 | UKDEJS250469 | likely_same | 0.0 | 0.8435 |

  Two of these (`UKDEJS014531`, `UKDEJS987250`) have LightGlue actively
  confirming the wrong candidate (`likely_same`), not just failing to
  object — the same pattern the first report already flagged.

- **Cost: 7 true positives demoted REVIEW→UNKNOWN** — up from the 4/102
  reported in the first pass, because the corrected stack has a larger true
  REVIEW population (68 vs. 44) exposed to a possible LightGlue demotion in
  the first place. Same mechanism, bigger sample, real cost:

  | Query animal | score | gap | `lightglue_zone` |
  |---|---|---|---|
  | UKDEJS211897 | 0.8651 | 0.0101 | likely_different |
  | UKDEJS280448 | 0.8081 | 0.0402 | likely_different |
  | UKDEJS725951 | 0.8718 | 0.0072 | likely_different |
  | UKDEJS791332 | 0.8436 | 0.0202 | likely_different |
  | UKDEOT336574 | 0.7834 | 0.0136 | likely_different |
  | UKDESH004551 | 0.81 | 0.0003 | likely_different |
  | UKDESH010833 | 0.8246 | 0.0674 | likely_different |

  Two of these (`UKDEJS211897`, `UKDESH004551`) are cases the attribute
  layer had *just fixed* (see the 6-case list above) — LightGlue then
  independently pulled them back down to UNKNOWN. The signals aren't
  additive-only; they can partially cancel on the same case.

### Direct answer to "does the attribute nudge catch any of the 5 survivors from the first report"

Of the first report's 5 wrong-animal cases that survived score+gap+LightGlue
(`UKDEGR827871`, `UKDEJS014531`, `UKDEJS155191`, `UKDEJS698982`,
`UKDEJS987250`) — under the fully-corrected stack, **2 are fixed**
(`UKDEJS155191`, `UKDEJS698982`, both via attribute agreement reordering the
top-1 pick to the correct animal) and **3 persist**
(`UKDEGR827871`, `UKDEJS014531`, `UKDEJS987250` — the same 3 in the table
above). The comparison is apples-to-oranges in one sense — the underlying
population changed once the aggregation bug was fixed — but the 3 remaining
codes are the same physical failure mode either way: an impostor animal
whose embedding score, color/horn attributes, AND LightGlue keypoint zone
(2 of 3 are `likely_same`) all agree with the query. No signal currently in
the stack has independent evidence against these 3.

### This is the real, complete "how good is it right now" answer

**23.6% of genuinely-unseen queries get a hands-free MATCH. 61.5% get the
correct animal put in front of an officer (MATCH or REVIEW). Of everything
shown to an officer, 97.1% is the right animal — the other 2.9% (3 cases)
needs a human's own judgment to reject, same as the first report already
found for the (smaller, pre-correction) REVIEW population. 5.6% of correct
matches (9/161) never reach a human at all, silently reading as "not
registered"; 7 of those 9 are LightGlue's own false calls on genuinely
correct matches, not a limitation of the embedding itself. 7.5% of queries
(12/161) never get to the matching stage because the query photo itself
failed a quality/detection gate.**

The muzzle-duplication caveat from the first report still applies
unchanged — this remains a lower bound on true 3-independent-photo
registration, not an exact prediction of it — but the earlier report's
specific "the strict-recall number is dragged down mainly by gap
suppression from the registration compromise" explanation was wrong. It's
now clear that most of that drag was a test-script bug that has nothing to
do with the registration compromise at all, and the real
attribute/LightGlue layers' costs and benefits are smaller and more
precisely bounded than the first pass suggested. A genuine, quantified gap
remains: **3/161 (1.9%) wrong-animal cases reach an officer with no signal
objecting, and 7/161 (4.3%) correct matches are silently lost specifically
to LightGlue's own false "likely_different" calls** — both real, both
small, and both the actual target for anything built next.


## Step 2: front/body-photo similarity as a fourth demote-only signal — tested, and NOT worth building

The full-stack result above leaves a small, real gap: 3/161 wrong-animal
cases reach REVIEW with no signal currently objecting, and separately
LightGlue costs 7/161 true positives. Before writing any production code,
tested whether a front-photo similarity signal — same demote-only pattern
as color/horn attributes and LightGlue, same trigger zone — could close
either gap. It cannot, and this was checked with real numbers before
building anything, not assumed.

**Feasibility: yes, mechanically trivial, confirmed by reading the actual
code before running anything.** `pipeline/muzzle.py`'s `embed_batch(images,
model, device)` is a generic function — raw image bytes in
(`preprocess_batch`: PIL decode → 518×518 resize → ImageNet normalize),
`(B, 256)` unit-norm `GodhaarModel` embeddings out. Nothing about it is
muzzle-specific; it has no crop/detection step baked in. Feeding it
`front1.jpg`/`front2.jpg` instead of a muzzle crop is a valid call through
the exact same code path — no new model, no retraining, no new
infrastructure.

**Calibration, before building anything:** loaded the real checkpoint
(`d:\Group Projects\Godhaar\Wildlifeor_adityaest_top1.pt`, the same one
used earlier this session for the direct cosine-similarity check) and
embedded every registered animal's `front2.jpg` (gallery side) and every
query's `front1.jpg` (query side — literally the same photo the real
leave-one-out `/search` calls already sent, so this measures what
production would actually see, not a synthetic setup). Two real
distributions, script `front_similarity_test.py` (scratchpad):

- **SAME-animal** (genuine pairs, n=161): `cos(query front1, own front2)` —
  mean **0.8727**, min **0.5658**, 5th-percentile **0.6854**.
- **DIFFERENT-animal** (hardest real negatives — the 41 `WRONG_ANIMAL` cases
  from the full-stack replay, n=41): `cos(query front1, wrongly-matched
  candidate's front2)` — mean **0.7501**, max **0.9112**.

**The two distributions overlap almost completely — this signal does not
discriminate identity for front/body photos.** Every single one of the 41
wrong-animal cases (100%) scores above the lowest genuine same-animal score
in the entire dataset (0.5658). There is no threshold that catches any real
impostor without also being low enough to never fire — a threshold at the
genuine-pair minimum catches 0/41 impostors by construction. Sweeping
looser thresholds confirms it's not just a bad cutoff choice, it's a
missing signal — false-positive risk on genuine pairs tracks catch rate on
impostors almost 1:1, the signature of no real separation, not a tuning
problem:

| Threshold (percentile of genuine distribution) | Impostors caught | Genuine pairs falsely flagged |
|---|---|---|
| 0th (0.5658) | 0/41 | 0/161 |
| 1st (0.5807) | 1/41 | 1/161 |
| 5th (0.6854) | 12/41 | 8/161 |
| 10th (0.7361) | 19/41 | 16/161 |
| 20th (0.7893) | 26/41 | 32/161 |

**Directly on the 3 remaining wrong-animal cases from Step 1** (the actual
target): `UKDEGR827871` scores 0.7647, `UKDEJS014531` scores 0.6735,
`UKDEJS987250` scores 0.6834 — all three sit at or above the genuine
distribution's 5th-to-95th-percentile bulk, indistinguishable from a normal
same-animal pair on this signal. None of them would be caught by any
threshold that doesn't also risk a meaningful fraction of all genuine
same-animal pairs in the ambiguous zone.

**Why it doesn't work, not just that it doesn't:** the checkpoint's own
saved metrics (`genuine_mean=0.7054, impostor_mean=0.0064, gap=0.699`) show
excellent separation — but that gap was earned by ArcFace fine-tuning
specifically on **muzzle crops**. The DINOv2 backbone underneath is generic,
but the projection head's decision boundary was trained to discriminate
muzzle texture/pattern, not whole-animal front-on appearance. Two more
effects compound against it for this specific use: front/body photos of
different animals from the same breed and region look broadly similar
(pose, background, coat pattern class), inflating cross-animal similarity,
while `front1` vs `front2` of the *same* animal can differ enough in
angle/lighting/pose to pull genuine similarity down — both push the two
distributions toward each other instead of apart.

**Conclusion: do not build this.** Not "needs tuning" — the ROC-like sweep
above shows no operating point exists where catch rate meaningfully
exceeds false-demotion rate on true positives, and the 3 specific cases
this was meant to catch score squarely inside the normal genuine-pair
range. Building the demote-only wiring (a real, cheap task, same shape as
`applyLightglueDisagreement`) would add a fourth signal that helps nothing
and risks new true-positive loss on top of LightGlue's existing 7. The
remaining 3/161 gap from Step 1 stays open; closing it would need a signal
actually trained to discriminate identity from the photo type in question —
this checkpoint is not that signal for front/body photos, and no amount of
threshold tuning changes that.


## Cheap feasibility check before building anything: horn/eye/ear geometry, and a narrower look at LightGlue's cost

Two follow-ups to the Step 1/Step 2 findings above, both checked before
writing any new code — same discipline as Step 2 (calibrate on real data,
report honestly, only build if the number justifies it).

### 1-2. Is there any real landmark geometry anywhere in the current pipeline? No.

**`horn_shape` (`pipeline/morphology.py`) is a pure classical-CV silhouette
heuristic, not a keypoint model, and exposes no reusable geometry.** Read
the actual code rather than assuming: `RuleBasedMorphologyExtractor.extract()`
takes `crop_cattle`'s whole-animal crop, cuts a fixed top-35% "head band"
(`HEAD_BAND_FRACTION`, not detected — a constant fraction of the crop),
runs Canny edge detection + finds the largest contour in that band, and
derives two categorical outputs from it: `has_horns` (is there a protrusion
above a pixel-fraction threshold) and `horn_shape` (is that protrusion's
edge straighter or more curved than a threshold, via `cv2.fitLine` residual
on the contour points). The module's own docstring says this directly:
*"There is no labeled horn dataset and no trained keypoint/shape model
anywhere in this project (checked both this repo's bundled wildlife/ and
the full Godhaar/Wildlife source — neither has one)."* The only
"geometry" that exists even internally (`top_y`, `horn_px`, the fitted-line
residuals) is a coarse silhouette measurement scoped to classifying one
enum value, never returned, and not eye/ear/horn coordinates in any
reusable sense.

**Nothing on the front-photo path detects eyes or ears, coarsely or
otherwise.** Grepped the whole repo for `eye|ear_|keypoint|landmark|facial`
— every real hit is either LightGlue/DISK's generic image keypoints
(muzzle-matching only, not facial landmarks — already documented above)
or a plain-English comment (`roi.py`: *"Top 20% (often contains
background, sky, ears, horns)"* — a fixed-fraction crop region to
*exclude*, not a detected ear). `crop_cattle` (`pipeline/yolo_crop.py`) is
a stock COCO YOLO (`yolov8s.pt`) doing whole-animal bounding boxes only —
one box per animal, the same detector already documented elsewhere in this
file, with no facial-part output of any kind.

**Conclusion: no real landmark coordinates exist anywhere in this pipeline
to calibrate against — the Step 3 calibration this task asked for cannot
be run, there is nothing to feed it.** This is a "don't build the
calibration test" finding by inspection, not a negative test result like
Step 2's front-similarity check — there was no live signal available to
even measure.

**Honest cost estimate for building one (Step 4):** this would be a new
model, not a config change or a reused embedding. Concretely: (a) no
existing cattle facial-landmark dataset is a standard off-the-shelf
resource the way COCO or ImageNet are — human/dog/cat facial-landmark
datasets exist, cattle-specific ones don't, so labeled data collection
starts from zero; (b) keypoint models are typically more annotation-hungry
per image than classification (each image needs multiple precise point
labels, not one), so a usable eye/ear/horn keypoint regressor likely needs
several hundred to low-thousands of hand-annotated photos across breeds,
angles, and lighting — a real annotation-tooling + labeling-labor project,
not an afternoon; (c) `GodhaarModel`'s architecture (DINOv2 → GeM pool →
projection head, trained for ArcFace metric learning) has no keypoint/
heatmap output head — this needs a genuinely different model design and
training loop, evaluated with different metrics (PCK, not top-1/recall);
(d) separately, Step 2 already showed that even a *well-trained*,
muzzle-specific embedding model didn't transfer to a different photo type
(front/body) with zero extra work — there's no guarantee a coarse
inter-eye/ear-span ratio would be individually discriminative even once
built, since that kind of gross anatomical proportion is more a function
of breed/age/pose than individual identity. Realistic estimate: weeks of
data collection, annotation, and training, not a cheap addition — and the
payoff is unproven until that investment is made. Not recommended to start
without a stronger reason than closing this specific 3/161 gap.

### 5. LightGlue's true-positive cost — a small, real, partial win exists; most of the cost doesn't have one

Pulled `lightglue_num_matches` (the raw keypoint-match count the
`likely_different`/`ambiguous`/`likely_same` zones are computed from —
`pipeline/lightglue_verify.py`'s `classify_zone()`) for every leave-one-out
search where LightGlue actually ran, split by same-animal vs
different-animal, and compared against the zone boundaries currently
shipped (`LIKELY_DIFFERENT_MAX = 140`, `LIKELY_SAME_MIN = 150`):

| Zone (current boundaries) | SAME-animal (n=96) | DIFFERENT-animal (n=41) |
|---|---|---|
| `< 140` (likely_different — triggers demotion) | 9 | 38 |
| `140–150` (ambiguous — no-op) | 2 | 0 |
| `>= 150` (likely_same — no-op today) | 85 | 3 |

**Two things worth separating here:**

**(a) The bulk of the signal genuinely holds up at this larger scale.** 85/96
(88.5%) of same-animal pairs land at `>= 150`, and 38/41 (92.7%) of
different-animal pairs land at `< 140` — directionally the same clean
separation the original 51-pair/9-10-animal calibration found (see
`lightglue_verify.py`'s own comment: *"28 clean FPs num_matches 28..139,
23 clean TPs 150..833, zero overlap"*), just not perfectly zero-overlap
once tested on ~3x more animals. Precision of the demotion trigger itself,
measured directly: **38/47 = 80.9%** of everything that lands in the `<140`
zone is a genuine impostor — the other 19.1% (9 cases) is exactly Step 1's
LightGlue cost.

**(b) Within that 9-vs-38 population, a small, real, boundary-only win
exists.** Sorted by `num_matches`: the 38 real impostor cases top out at
**117**. Two of the 9 same-animal cases sit clearly above that —
`UKDEJS725951` at **119** and `UKDESH010833` at **136** — with zero overlap
between them and the entire impostor distribution. Sweeping the boundary
down from 140 confirms this precisely:

| Candidate `LIKELY_DIFFERENT_MAX` | True positives saved | Real catches lost |
|---|---|---|
| 140 (current) | 0/9 | 0/38 |
| 130 | 2/9 | 0/38 |
| **118–119** | **3/9** | **0/38** |
| 115 | 3/9 | 1/38 |
| 110 | 3/9 | 3/38 |
| 100 | 3/9 | 4/38 |
| 80 | 4/9 | 8/38 |
| 60 | 4/9 | 18/38 |
| 40 | 6/9 | 27/38 |

Lowering `LIKELY_DIFFERENT_MAX` from 140 to **~118** recovers **2 of the 7
visible true-positive costs from Step 1** (`UKDEJS725951`, `UKDESH010833`
— a 3rd, `UKDEOT691947` at 137, is also pulled out of the zone but was
already `UNKNOWN` regardless of LightGlue, from the attribute layer alone,
so it has no visible effect either way) **at zero measured cost** — no real
impostor in this 161-animal set has a `num_matches` anywhere near that
range. Below ~110 the trade reverses fast: every further step down costs
more real catches than it recovers true positives (e.g. 100→80 trades 1
saved TP for 4 more lost catches). **This is a genuine, cheap,
config-only recommendation** (one constant in `lightglue_verify.py`, no
new code) that recovers ~29% of LightGlue's measured true-positive cost
for free — but it is not a fix for the other ~71% (5–7 of the 7 cases):
their `num_matches` values (34, 36, 39, 50, 95) sit thoroughly inside the
impostor distribution's own range, so no single-variable threshold on this
signal can separate them without also losing real catches.

**Caveat, stated with the same honesty as the original calibration's own
comment about its small sample:** the "clean gap" this recommendation
rests on is thin — 2 same-animal points (119, 136) against a same-population
impostor ceiling of 117, a 2-count margin. That's a real, measured gap in
this specific 161-animal dataset, not a coin flip, but it's also not a
large-sample, wide-margin result — a different or larger dataset could
plausibly shift it. Worth taking (it's free and reversible, a one-line
constant), not worth treating as a fully validated new threshold the way
`matchThreshold`/`reviewThreshold` are.
