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

## Horn/ear morphology — a new signal for the weak color gate, v1 is an unvalidated heuristic

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

**Deliberately scoped to this service only.** go-apiserver relays this
`detail` string to the app as-is, so the app already surfaces the longer
combined message — but nobody has reshaped it into a structured per-photo
list on either go-apiserver or the app yet; the app's error handling still
treats the whole thing as one opaque string (unlike the existing
client-side blur guard, which already lists bad photo numbers cleanly). If
that's wanted, it needs matching changes in go-apiserver's response and
`src/api/animals.ts` — not done here.

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
