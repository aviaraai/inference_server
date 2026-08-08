# Godhaar Inference Server — working notes

FastAPI + PyTorch (CUDA) service that detects, embeds, and matches cattle
muzzles. It is called only by the API server (never the app directly), returns
raw similarity scores, and knows nothing about farmers/ownership. Endpoints:
`POST /register` (201), `POST /search`, `GET /health`. Port 9050.

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

**Status: local working-tree change only — NOT committed, NOT deployed.** The
running GPU server is unaffected until someone pulls this and restarts the
inference container. **This fix removes the catastrophic case but has NOT been
confirmed to fully fix matching** — see "Still open" below.

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
| Several photos failing for *different* reasons (422) | `POOR_IMAGE_QUALITY` | same; per-photo codes survive on each entry |
| Front photos disagree on body colour (422) | `BODY_COLOR_INCONSISTENT` | same, slots `front_1`/`front_2` |
| No muzzle-colour majority (422) | `MUZZLE_COLOR_INCONSISTENT` | same, slots `muzzle_1..3` |

**What must NOT get an envelope**, and why the handler is opt-in rather than
blanket: failures that are not a verdict about this animal or these photos —
FastAPI request validation, wrong image count, malformed `candidates` JSON,
FAISS errors — stay plain `HTTPException`. Those genuinely *are* the two
services being out of step, and go-apiserver is right to classify them as
contract/transport faults. Stamping a code onto them would surface a renamed
form field to a farmer as "retake your photos".

Two known gaps, both on the go-apiserver side, neither breaking:
- `decodeTypedDetail`'s switch omits the two colour codes, so `detail.failures`
  is dropped for those (verdict and user copy are still correct; only the
  `failed_images` hint is lost). Adding the two cases to that switch is the fix.
- `RECAPTURE_MULTI_CATTLE` is flattened into `NO_ANIMAL_DETECTED` because
  go-apiserver has no code for "several animals in frame", and an unrecognised
  code there degrades to a generic service error. The real `det_status` survives
  in `failures[].reason`. Adding a `MULTI_CATTLE` code to both sides is the fix.

## Decision thresholds live in the API SERVER, not here

This service returns raw scores only. MATCH/REVIEW/UNKNOWN and the 3 km GPS
candidate filter are the API server's job (`decide()`: MATCH ≥ 0.82 with
rank1−rank2 gap ≥ 0.08; REVIEW ≥ 0.72; else UNKNOWN). A single candidate in
range means gap = 0, so it can never be a clean MATCH — a score ≥ 0.72 becomes
REVIEW, which the app still surfaces as a result.

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
