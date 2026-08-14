"""
schema.py — Pydantic response models for the inference server API.

Request parameters use Form() + File() directly in route signatures.
These models define the response shape only.
"""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ── Shared ────────────────────────────────────────────────────────────────────

class VersionInfo(BaseModel):
    model: str = Field(..., description="GodhaarModel version identifier")
    faiss: Optional[str] = Field(None, description="FAISS index date label")
    embedding: Optional[str] = Field(None, description="Embedding contract version")


class ColorResult(BaseModel):
    label: str = Field(..., description="Color enum label (e.g. BLACK, PINK, UNKNOWN)")
    confidence: float = Field(..., description="Classifier confidence 0.0–1.0")


class ExtractedColors(BaseModel):
    body: ColorResult
    muzzle: ColorResult


HORN_SHAPE_DESCRIPTION = (
    "One of pipeline.morphology.HornShape (STRAIGHT, CURVED, UNKNOWN), or "
    "None if no horn is confirmed visible in the photo(s). None does NOT "
    "mean 'confirmed hornless' — a genuinely polled animal and a "
    "backward/occluded horn look identical to this heuristic, and any "
    "non-reading status (bad image, no animal detected, no clear "
    "silhouette, or — register only — the 2 front photos disagreeing) "
    "also collapses to None here. See pipeline/morphology.py."
)


# ── Errors ────────────────────────────────────────────────────────────────────
#
# Every deliberate 4xx verdict this server reaches is returned as
#
#     {"error_code": "<ErrorCode>", "detail": {...}}
#
# and NOT as FastAPI's bare {"detail": ...}. The envelope exists because the
# API server keys its user-facing response off `error_code` alone: a body
# without one is classified there as a contract violation ("the two services
# are out of step") and shown to the farmer as a generic service error. So a
# duplicate animal or a blurry photo returned without this envelope is not a
# cosmetic difference — the verdict is thrown away.
#
# The flip side is the reason this is opt-in rather than a blanket handler:
# failures that are NOT a verdict about the animal or the photos — FastAPI's
# own request validation, a wrong image count, malformed candidates JSON — must
# keep answering with the bare {"detail": ...}. Those genuinely ARE the two
# services being out of step, and the API server is right to treat them that
# way. See errors.py for the raise sites.


class ErrorCode(str, Enum):
    """Machine-readable verdicts. One contract, two implementations: this enum
    and `domainCodes` in the API server's internal/inference/errors.go must be
    changed together.

    The API server degrades a code it does not recognise to a contract failure
    rather than guessing at it, so a new code can be deployed here first and
    taught to the API server afterwards — but until it is, the verdict does not
    reach the user.
    """

    # Register only: this muzzle is already in the index.
    DUPLICATE_ANIMAL = "DUPLICATE_ANIMAL"

    # YOLO found nothing to crop.
    NO_ANIMAL_DETECTED = "NO_ANIMAL_DETECTED"

    # YOLO found several animals and none of them stands out as the subject.
    # Distinct from NO_ANIMAL_DETECTED because the advice is the opposite:
    # there is nothing wrong with the photo, the officer just has to isolate
    # the animal they mean. Telling them to "step back so the whole animal is
    # in frame" makes a multi-cattle frame worse.
    MULTI_CATTLE = "MULTI_CATTLE"

    # Single-image quality failures, split by cause because each one implies
    # different advice to whoever is holding the phone.
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    IMAGE_BAD_EXPOSURE = "IMAGE_BAD_EXPOSURE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_UNREADABLE = "IMAGE_UNREADABLE"

    # Cross-image disagreement: each photo is fine on its own but together they
    # cannot describe one animal.
    BODY_COLOR_INCONSISTENT = "BODY_COLOR_INCONSISTENT"
    MUZZLE_COLOR_INCONSISTENT = "MUZZLE_COLOR_INCONSISTENT"

    # Umbrella for a set of images that failed for more than one distinct
    # reason; the per-image codes survive in the detail.
    POOR_IMAGE_QUALITY = "POOR_IMAGE_QUALITY"


class ImageFailure(BaseModel):
    """One photo's rejection. `slot` is what makes this worth sending: the app
    marks that specific image for retaking instead of asking for all five.
    """

    slot: str = Field(..., description="Which upload failed: muzzle_1..3, front_1..2, muzzle, front")
    stage: str = Field(..., description="Where it failed: decode, quality, detection, crop_quality, color")
    error_code: ErrorCode = Field(..., description="Per-image code; may differ from the envelope's code")
    reason: str = Field(..., description="Raw pipeline reason string, for logs — never shown to a user")


class ColorReading(BaseModel):
    """What one photo actually read, on a colour-disagreement rejection.

    This is the context that makes the verdict actionable instead of merely
    negative. "Your photos disagree, retake them" invites the officer to take
    the same two photos again; "front_1 read BLACK, front_2 read WHITE" tells
    them the far more likely truth — that the two photos are of different
    animals — which is something they can actually fix.
    """

    slot: str
    label: str = Field(..., description="Colour label this photo produced")
    confidence: float = Field(..., description="The winning label's share of the sampled region, 0.0–1.0")


class ImageQualityDetail(BaseModel):
    """Detail payload for every image-quality and colour-consistency code."""

    message: str = Field(..., description="Internal summary. The API server writes its own user-facing copy.")
    failures: list[ImageFailure]
    readings: list[ColorReading] = Field(
        default_factory=list,
        description="Colour codes only: the per-photo readings that disagreed. Empty for quality failures.",
    )


class DuplicateDetail(BaseModel):
    """Detail payload for DUPLICATE_ANIMAL.

    `matched_faiss_id` is the field that matters: this server knows only FAISS
    ids, and the API server is the only side that can turn one back into the
    godhaar_id the officer needs in order to go and look at the registration
    they are duplicating.
    """

    matched_faiss_id: int
    top_score: float
    body_color: str = Field(..., description="Colour freshly extracted from this request, which matched the stored one")
    muzzle_color: str


class ErrorEnvelope(BaseModel):
    """The wire shape of every domain rejection. Declared for the OpenAPI docs
    and as the single written-down description of the contract; the responses
    themselves are built in errors.py.
    """

    error_code: ErrorCode
    detail: dict[str, Any]


# ── Candidate input (register duplicate-check AND search) ─────────────────────

class CandidateInfo(BaseModel):
    """A nearby cattle candidate sent by the API server, for duplicate
    checking (/register) or for echoing stored data back on each match
    (/search). `horn_shape` is optional so a caller that doesn't have it
    yet can omit it — it defaults to None, not a fabricated value.
    """
    faiss_id: int = Field(..., description="FAISS integer ID of the stored embedding")
    body_color: str = Field(..., description="Stored body color label (e.g. BLACK)")
    muzzle_color: str = Field(..., description="Stored muzzle color label (e.g. PINK)")
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)
    # NOT a price. This animal's tag_no -- see the `tag_no` parameter on
    # /register in main.py. Optional so a candidate registered before tag_no
    # existed can still be sent without one; the duplicate-check veto below
    # only fires when BOTH sides have a tag to compare.
    tag_no: Optional[str] = Field(
        None, description="This candidate's stored tag_no, echoed back so it can be compared against the query's own tag_no."
    )


# ── Match ─────────────────────────────────────────────────────────────────────

class MatchCandidate(BaseModel):
    faiss_id: int = Field(..., description="FAISS integer ID for this embedding")
    score: float = Field(..., description="Cosine similarity score")
    rank: int = Field(..., description="1-indexed rank among returned candidates")
    gap: float = Field(..., description="Score delta to the next candidate (0.0 for last)")
    body_color: Optional[str] = Field(
        None, description="This candidate's stored body color, echoed back from the request's candidates list, if provided."
    )
    muzzle_color: Optional[str] = Field(
        None, description="This candidate's stored muzzle color, echoed back from the request's candidates list, if provided."
    )
    horn_shape: Optional[str] = Field(
        None, description="This candidate's stored horn shape, echoed back so it can be compared against the query's own `horn_shape` field. " + HORN_SHAPE_DESCRIPTION
    )


# ── Register ──────────────────────────────────────────────────────────────────
class RegisterResponse(BaseModel):
    status: str = Field("success", description="Registration status")
    embedding_ids: list[int] = Field(
        ..., description="FAISS integer IDs for the stored embeddings"
    )
    extracted_colors: ExtractedColors
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)
    potential_matches: list[MatchCandidate] = Field(
        default_factory=list,
        description="Top matches against candidate_ids (empty if no candidates provided)",
    )
    versions: VersionInfo
    registered_at: str = Field(..., description="ISO 8601 UTC timestamp")


# ── Search ────────────────────────────────────────────────────────────────────

class SearchResponse(BaseModel):
    request_id: str = Field(..., description="UUID for request tracing")
    query_colors: ExtractedColors
    horn_shape: Optional[str] = Field(None, description=HORN_SHAPE_DESCRIPTION)
    top_matches: list[MatchCandidate]

    # ── Fusion tiebreaker (additive, inert) ──────────────────────────────────
    # Describes query vs. top_matches[0] specifically, not any individual
    # match row — hence top-level, not on MatchCandidate. Populated only when
    # the top-1 embedding score was ambiguous (near go-apiserver's
    # MATCH/REVIEW/GAP thresholds) AND a cached crop existed for the top
    # candidate; otherwise lightglue_checked is False and the other two stay
    # None. Nothing here is consumed by this server or by go-apiserver today
    # — see CLAUDE.md.
    lightglue_checked: bool = Field(
        False, description="Whether the LightGlue keypoint-matching tiebreaker actually ran for this search."
    )
    lightglue_num_matches: Optional[int] = Field(
        None, description="Raw LightGlue match count between the query crop and top_matches[0]'s cached crop. None if lightglue_checked is False."
    )
    lightglue_zone: Optional[Literal["likely_same", "likely_different", "ambiguous"]] = Field(
        None, description="Calibrated zone for lightglue_num_matches (see pipeline/lightglue_verify.py). None if lightglue_checked is False."
    )

    versions: VersionInfo


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field("ok")
    model_loaded: bool
    faiss_size: int = Field(..., description="Total vectors in FAISS index")
    gpu_available: bool
    model_version: str
    color_extractor_available: bool
    morphology_extractor_available: bool = Field(
        True, description="Always true — the rule-based morphology extractor has no external dependency to fail."
    )
