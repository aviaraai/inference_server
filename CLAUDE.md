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
