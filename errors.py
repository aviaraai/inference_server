"""
errors.py — domain rejections and the error envelope they travel in.

A "domain" rejection is a verdict this server was asked for and reached: this
muzzle is a duplicate, that photo is too blurry to embed. It is the only kind
of failure whose meaning is safe to explain to the officer holding the phone,
and the only kind that carries an `error_code` (see schema.ErrorCode).

Everything else — a missing form field, three muzzle images where the route
wants three and got two, malformed candidates JSON, FAISS falling over — stays
a plain HTTPException. Those answer with FastAPI's bare {"detail": ...}, which
the API server classifies as a contract or transport fault. That is the correct
outcome for them and the reason this module is opt-in: if a blanket handler
stamped an error_code onto every failure, a renamed form field would reach a
farmer as "retake your photos".

The classifiers below are the seam between the pipeline's free-form reason
strings and the fixed code vocabulary. They live here, not at the raise sites,
so that the string formats produced in pipeline/quality.py and
pipeline/yolo_crop.py are matched in exactly one place.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schema import (
    ColorReading,
    DuplicateDetail,
    ErrorCode,
    ImageFailure,
    ImageQualityDetail,
)


class DomainError(HTTPException):
    """A verdict, rendered as {"error_code": ..., "detail": {...}}.

    Subclasses HTTPException on purpose. Raising it from inside
    asyncio.to_thread works exactly as the plain one does (to_thread re-raises
    on the awaiting coroutine), and if the handler below were ever left
    unregistered the request would still fail as a sane 4xx with the detail
    intact, rather than becoming a 500.

    `detail` is always an object, never a bare string: the API server parses it
    into a typed shape per code, and a string there would silently degrade a
    good verdict into "duplicate rejection carried no usable matched_faiss_id".
    """

    def __init__(self, status_code: int, error_code: ErrorCode, detail: BaseModel | dict):
        payload = detail.model_dump(mode="json") if isinstance(detail, BaseModel) else detail
        super().__init__(status_code=status_code, detail=payload)
        self.error_code = error_code

    def envelope(self) -> dict:
        return {"error_code": self.error_code.value, "detail": self.detail}


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Registered on the app in main.py.

    Starlette resolves handlers by walking type(exc).__mro__ and taking the
    first match, so this wins over the built-in HTTPException handler for
    DomainError while leaving every plain HTTPException untouched.
    """
    return JSONResponse(status_code=exc.status_code, content=exc.envelope())


# ── Classifiers ───────────────────────────────────────────────────────────────


def classify_quality_reason(reason: str) -> ErrorCode:
    """Map a quality_check / quality_check_cv2 reason to a code.

    The reasons are formatted with the measured value attached
    ("bad_quality blur=41.23"), so this matches on the metric name rather than
    the whole string. An unrecognised reason falls back to the umbrella code:
    still a real photo complaint, just without the specific advice — better
    than a code the API server has never heard of, which would surface as a
    service fault and tell the officer nothing.
    """
    if "cannot_decode_image" in reason or "empty_image" in reason:
        return ErrorCode.IMAGE_UNREADABLE
    if "short=" in reason:
        return ErrorCode.IMAGE_TOO_SMALL
    if "blur=" in reason:
        return ErrorCode.IMAGE_TOO_BLURRY
    if "exposure=" in reason:
        return ErrorCode.IMAGE_BAD_EXPOSURE
    return ErrorCode.POOR_IMAGE_QUALITY


def classify_detection_status(status: str) -> ErrorCode:
    """Map a crop_cattle det_status to a code.

    The two outcomes need opposite advice, which is why they are separate
    codes: NO_DETECTION means get more of the animal into frame, MULTI_CATTLE
    means get less of everything else into it. Collapsing them — as this
    briefly did while go-apiserver had no MULTI_CATTLE code — hands a goshala
    officer the one instruction guaranteed to make their photo worse.
    """
    if "MULTI_CATTLE" in status:
        return ErrorCode.MULTI_CATTLE
    return ErrorCode.NO_ANIMAL_DETECTED


def _umbrella_code(failures: list[ImageFailure]) -> ErrorCode:
    """One distinct cause → name it. Several → POOR_IMAGE_QUALITY.

    The per-image codes are not lost when the umbrella is used; they stay on
    each entry in `failures`, which is why the API server can still mark the
    right photos even when the envelope's own code is the generic one.
    """
    distinct = {f.error_code for f in failures}
    if len(distinct) == 1:
        return next(iter(distinct))
    return ErrorCode.POOR_IMAGE_QUALITY


# ── Constructors ──────────────────────────────────────────────────────────────


def image_quality_error(failures: list[ImageFailure]) -> DomainError:
    """422 for one or more unusable photos, code derived from the causes."""
    summary = "; ".join(f"{f.slot}: {f.reason}" for f in failures)
    return DomainError(
        status_code=422,
        error_code=_umbrella_code(failures),
        detail=ImageQualityDetail(message=summary, failures=failures),
    )


def color_readings(slot_prefix: str, colors: list[dict]) -> list[ColorReading]:
    """Turn the extractor's raw {label, confidence} dicts into readings,
    numbering the slots the way the uploads were numbered (front_1, muzzle_2).
    """
    return [
        ColorReading(
            slot=f"{slot_prefix}_{i}",
            label=c["label"],
            confidence=round(float(c["confidence"]), 4),
        )
        for i, c in enumerate(colors, 1)
    ]


def color_inconsistency_error(
    error_code: ErrorCode, readings: list[ColorReading], summary: str
) -> DomainError:
    """422 for photos that are individually fine but disagree with each other.

    Carried in the same ImageQualityDetail shape as a quality failure so the
    app has one branch for "these images need retaking", not two — plus the
    `readings` that explain the disagreement. Every contributing slot is
    listed because no single one of them is at fault: the disagreement is the
    defect, and the officer needs to see both sides of it to work out which
    photo is the odd one out.
    """
    failures = [
        ImageFailure(
            slot=r.slot,
            stage="color",
            error_code=error_code,
            reason=f"read {r.label} at {r.confidence:.2f}",
        )
        for r in readings
    ]
    return DomainError(
        status_code=422,
        error_code=error_code,
        detail=ImageQualityDetail(
            message=summary, failures=failures, readings=readings
        ),
    )


def duplicate_animal_error(
    matched_faiss_id: int, top_score: float, body_color: str, muzzle_color: str
) -> DomainError:
    """409, nothing written to FAISS."""
    return DomainError(
        status_code=409,
        error_code=ErrorCode.DUPLICATE_ANIMAL,
        detail=DuplicateDetail(
            matched_faiss_id=matched_faiss_id,
            top_score=top_score,
            body_color=body_color,
            muzzle_color=muzzle_color,
        ),
    )


def unreadable_image_error(slot: str, stage: str = "decode") -> DomainError:
    """422 for bytes OpenCV could not turn into pixels at all."""
    return image_quality_error(
        [
            ImageFailure(
                slot=slot,
                stage=stage,
                error_code=ErrorCode.IMAGE_UNREADABLE,
                reason="cannot_decode_image",
            )
        ]
    )
