"""pipeline/tests/test_rerank.py — pure-logic tests for the top-K LightGlue
re-ranking combination rule (pipeline/rerank.py). No GPU/model needed."""
from pipeline.rerank import rank_by_lightglue_evidence, rrf_combine


def test_rrf_combine_rewards_agreement():
    # Both signals agree this candidate is best -> highest combined score.
    best = rrf_combine(embedding_rank=1, lightglue_rank=1)
    worst = rrf_combine(embedding_rank=5, lightglue_rank=5)
    assert best > worst


def test_rrf_combine_missing_lightglue_rank_degrades_gracefully():
    # A candidate with no cached crop (lightglue_rank=None) should score
    # LOWER than an identical embedding rank WITH lightglue evidence backing
    # it up, but should NOT be penalized relative to its own embedding-only
    # position -- i.e. it still beats a worse-embedding-rank candidate.
    with_evidence = rrf_combine(embedding_rank=2, lightglue_rank=1)
    without_evidence = rrf_combine(embedding_rank=2, lightglue_rank=None)
    assert with_evidence > without_evidence

    worse_embedding_rank = rrf_combine(embedding_rank=4, lightglue_rank=None)
    assert without_evidence > worse_embedding_rank


def test_rrf_combine_can_promote_lower_embedding_rank():
    # The whole point of re-ranking: strong LightGlue evidence for a
    # lower-embedding-rank candidate can outscore a weak-LightGlue top
    # embedding candidate.
    weak_embedding_leader = rrf_combine(embedding_rank=1, lightglue_rank=20)
    strong_lightglue_challenger = rrf_combine(embedding_rank=2, lightglue_rank=1)
    assert strong_lightglue_challenger > weak_embedding_leader


def test_rank_by_lightglue_evidence_orders_by_ratio_then_count():
    candidates = [
        {"faiss_id": 1, "match_ratio": 0.10, "num_matches": 50},
        {"faiss_id": 2, "match_ratio": 0.30, "num_matches": 10},
        {"faiss_id": 3, "match_ratio": 0.30, "num_matches": 20},  # ties faiss_id=2 on ratio, wins on count
    ]
    ranks = rank_by_lightglue_evidence(candidates)
    assert ranks[3] == 1
    assert ranks[2] == 2
    assert ranks[1] == 3


def test_rank_by_lightglue_evidence_omits_candidates_with_no_ratio():
    candidates = [
        {"faiss_id": 1, "match_ratio": 0.5, "num_matches": 100},
        {"faiss_id": 2, "match_ratio": None, "num_matches": None},
    ]
    ranks = rank_by_lightglue_evidence(candidates)
    assert ranks == {1: 1}
    assert 2 not in ranks
