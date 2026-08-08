# `/register` and `/search` — API reference (with `horn_shape`)

Reflects `origin/inference_server` as of commit `a52d96b`. This service is a
pure ML microservice — it returns raw scores/classifications only. All
business logic (MATCH/REVIEW/NOT_REGISTERED decisions, GPS filtering, policy)
lives in the API server, not here.

---

## `horn_shape` — what it is, at a glance

Every endpoint below that used to return a full `morphology` object
(`has_horns`, `horn_shape`, `confidence`, `status`, `reason`) now returns a
**single field**: `horn_shape: string | null`.

| Value | Meaning |
|---|---|
| `"STRAIGHT"` | A horn was detected; its edge contour was close to a straight line. |
| `"CURVED"` | A horn was detected; its edge contour deviated notably from straight. |
| `"UNKNOWN"` | A horn was detected, but its shape couldn't be classified (too few contour points to fit a line reliably). |
| `null` | No horn confirmed visible **in this photo** — see the caveat below. Also what you get for any failed/ambiguous reading (bad image, no animal detected, no clear silhouette, or — register only — the two front photos disagreeing). |

**`null` is not "confirmed hornless."** A single front photo can't tell a
genuinely polled/dehorned animal (routine in Indian cattle) apart from an
animal whose horns just aren't visible from that angle (backward-curving,
occluded, poor lighting). Both cases produce `null`. Never treat `null` as
negative evidence.

**This is a return-only field, not a gate.** Unlike body/muzzle color, a bad
or inconsistent horn reading never blocks a registration or search with a
4xx — it just comes back as `null`.

**This is an unvalidated v1 heuristic**, not a trained model — see
[How it's computed](#how-horn_shape-is-computed) below. There is no labeled
horn dataset anywhere in this project.

---

## `POST /register`

Registers a cattle animal: embeds 3 muzzle photos, checks for duplicates
against caller-supplied candidates, stores in FAISS.

### Request — `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `muzzle_images` | file[] | yes, exactly 3 | |
| `front_images` | file[] | yes, exactly 2 | Also used to read `horn_shape`. |
| `candidates` | string (JSON) | yes | JSON-encoded list of `CandidateInfo`. Send `"[]"` if none nearby. |

`candidates` JSON shape (`CandidateInfo`):
```json
[
  {
    "faiss_id": 123,
    "body_color": "BLACK",
    "muzzle_color": "PINK",
    "horn_shape": "CURVED"
  }
]
```
`horn_shape` on a candidate is optional — omit it if the API server doesn't
have a stored value for that animal yet (defaults to `null`, not a
fabricated guess).

### Response — `201 Created` (`RegisterResponse`)

```json
{
  "status": "success",
  "embedding_ids": [4831, 4832, 4833],
  "extracted_colors": {
    "body":   { "label": "BLACK", "confidence": 0.91 },
    "muzzle": { "label": "PINK",  "confidence": 0.88 }
  },
  "horn_shape": "STRAIGHT",
  "potential_matches": [
    {
      "faiss_id": 4801,
      "score": 0.34,
      "rank": 1,
      "gap": 0.05,
      "body_color": "BLACK",
      "muzzle_color": "PINK",
      "horn_shape": "CURVED"
    }
  ],
  "versions": { "model": "v3.2", "faiss": "2026-08-01", "embedding": "v1" },
  "registered_at": "2026-08-06T09:14:22.101Z"
}
```

| Field | Notes |
|---|---|
| `horn_shape` | Fresh reading from this registration's own 2 `front_images` — see below. |
| `potential_matches[].horn_shape` | **Echoed back** from the `candidates` list you sent in — not recomputed. Lets you compare the new animal's `horn_shape` against a near-duplicate's stored value without a second lookup. |

### HTTP status codes

| Code | Meaning |
|---|---|
| `201` | Registered. |
| `409` | Duplicate muzzle (embedding similarity ≥ `DUPLICATE_THRESHOLD` **and** matching body + muzzle color). `horn_shape` plays no part in this check. |
| `422` | Bad input — wrong image count, quality failure, no animal detected, or unresolved color disagreement. **Never fires because of `horn_shape`.** |
| `500` | FAISS/internal failure. |

---

## `POST /search`

Embeds one muzzle + one front photo, ranks only the caller-supplied
candidates (no index-wide search — the API server does GPS/Supabase
filtering before calling this).

### Request — `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `muzzle` | file | yes | |
| `front` | file | yes | Used to read `horn_shape` for the query. |
| `top_k` | int | no (default 5) | |
| `candidates` | string (JSON) | yes, non-empty | Same `CandidateInfo` shape as `/register` — see above. |

### Response — `200 OK` (`SearchResponse`)

```json
{
  "request_id": "9f1c2b7a...",
  "query_colors": {
    "body":   { "label": "BLACK", "confidence": 0.89 },
    "muzzle": { "label": "PINK",  "confidence": 0.91 }
  },
  "horn_shape": "CURVED",
  "top_matches": [
    {
      "faiss_id": 4801,
      "score": 0.87,
      "rank": 1,
      "gap": 0.12,
      "body_color": "BLACK",
      "muzzle_color": "PINK",
      "horn_shape": "CURVED"
    }
  ],
  "versions": { "model": "v3.2", "faiss": "2026-08-01", "embedding": null }
}
```

| Field | Notes |
|---|---|
| `horn_shape` (top level) | Fresh reading from the query's own `front` photo. |
| `top_matches[].horn_shape` | **Echoed back** from the matching candidate's entry in the `candidates` you sent in — same value, not recomputed. Compare it against the top-level `horn_shape` yourself; this endpoint does no comparison math. |

### HTTP status codes

| Code | Meaning |
|---|---|
| `200` | Search ran (may still return zero matches). |
| `422` | Bad input — quality failure, no animal detected, empty `candidates`, or malformed `candidates` JSON. **Never fires because of `horn_shape`.** |

---

## How `horn_shape` is computed

Source: `pipeline/morphology.py`, `RuleBasedMorphologyExtractor` — classical
CV, not a trained model (no labeled horn dataset exists in this project).

1. Take the whole-animal crop of the **front** photo (`crop_cattle`, the same
   YOLO detector used elsewhere).
2. Treat the **top 35%** of that crop as the head/horn/ear band.
3. Find the largest edge contour in that band (Canny + dilate). Too small or
   absent → no reading (`null`).
4. Measure how far the contour protrudes above the band's base. Under ~3% of
   crop width → **no horn** (`null`).
5. If a protrusion is found → fit a straight line through its contour points
   and measure RMS deviation from that line, normalized by the horn's own
   extent. Below the straightness cutoff → `STRAIGHT`; above → `CURVED`; too
   few points to fit reliably → `UNKNOWN`.

### `/register` vs `/search` differ in how many photos feed this

- **`/search`**: one `front` photo → one direct reading. Whatever
  `RuleBasedMorphologyExtractor.extract()` returns is what you get.
- **`/register`**: **two** `front_images` → each gets its own reading, then
  `average_readings()` reconciles them:
  - both produced a reading and **agree** → that value.
  - both produced a reading and **disagree** → `null` (a categorical
    disagreement like STRAIGHT vs CURVED can't be averaged, and unlike color
    this never blocks registration with a 422 — it just comes back `null`).
  - only **one** photo produced a reading → that one's value is used.
  - **neither** produced a reading → `null`.

In both cases, only the final `horn_shape` string (or `null`) crosses the
wire — the internal confidence/status/reason used to reach that value never
leaves this service.
