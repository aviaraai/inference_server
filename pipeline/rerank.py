"""
pipeline/rerank.py — combine embedding rank + LightGlue keypoint evidence
into one ranking for /search's top-K candidates.

Promotes LightGlue from a demote-only veto on a single top-1 candidate (see
pipeline/lightglue_verify.py's original tiebreaker) to a signal used across
all top-K embedding candidates. Built and gated on
scratch_rerank_eval.py's Stage 1 measurement — see that script and
CLAUDE.md for the numbers this shipped on. Do not change RRF_C or the
combination shape without re-running that gate.

This module produces EVIDENCE ONLY — a per-candidate LightGlue rank/ratio
that main.py attaches to each MatchCandidate in the raw /search response.
It does NOT decide which candidate is correct or reorder `top_matches`:
that stays go-apiserver's job (this server returns raw scores only — see
CLAUDE.md's "Decision thresholds live in the API SERVER, not here").
"""

from __future__ import annotations

RRF_C = 60  # standard reciprocal-rank-fusion constant, matches scratch_rerank_eval.py


def rrf_combine(embedding_rank: int, lightglue_rank: int | None, c: int = RRF_C) -> float:
    """Reciprocal rank fusion of an embedding-score rank and a LightGlue
    match_ratio rank, both 1-indexed WITHIN the top-K set being considered.

    A missing lightglue_rank (no cached crop for that candidate --
    MUZZLE_CROP_CACHE_DIR only covers animals registered after that
    feature shipped) contributes 0 rather than the worst possible rank, so
    a crop-less candidate keeps its embedding-derived position relative to
    other crop-less candidates instead of being punished for a cache gap
    it had no control over -- this is the graceful-degradation requirement
    from the Stage 2 design.

    Not a weighted score sum on purpose: embedding cosine similarity and
    LightGlue's match_ratio live on unrelated scales (a 0.02 embedding-
    score gap and a 0.02 match_ratio gap mean nothing comparable), so
    combining ranks rather than raw scores avoids having to invent a
    conversion between them.
    """
    score = 1.0 / (c + embedding_rank)
    if lightglue_rank is not None:
        score += 1.0 / (c + lightglue_rank)
    return score


def rank_by_lightglue_evidence(
    candidates: list[dict],
) -> dict[int, int]:
    """candidates: list of {"faiss_id": int, "match_ratio": float | None,
    "num_matches": int | None} for the top-K set (one entry per faiss_id
    LightGlue was actually run against -- callers omit faiss_ids with no
    cached crop rather than passing None match_ratio for them, since those
    never got a LightGlue call at all).

    Returns {faiss_id: lightglue_rank}, 1-indexed by match_ratio
    descending (ties broken by num_matches descending). Candidates not
    present in the input dict simply have no key here -- callers pass
    `.get(faiss_id)` (-> None) into rrf_combine for those, which is exactly
    the graceful-degradation contract rrf_combine documents.
    """
    usable = [c for c in candidates if c.get("match_ratio") is not None]
    usable.sort(key=lambda c: (-c["match_ratio"], -(c.get("num_matches") or 0)))
    return {c["faiss_id"]: idx + 1 for idx, c in enumerate(usable)}
