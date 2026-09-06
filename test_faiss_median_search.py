"""
test_faiss_median_search.py — unit tests for FaissIndex.restricted_search's
multi-photo (median-aggregated) query path.

Covers: N=1 backward compatibility (median of one value is that value,
median_query_idx is always 0), N=3 median correctness against hand-computed
expected values, and median_query_idx exactness (for odd N, the median is
always one of the actual samples -- this asserts equality, not "closest").
"""

import asyncio
import unittest

import numpy as np

from faiss_index import FaissIndex


def _run(coro):
    return asyncio.run(coro)


class TestRestrictedSearchMedian(unittest.TestCase):
    def setUp(self):
        self.index = FaissIndex(embedding_dim=4)
        # Three orthogonal-ish unit vectors, easy to hand-verify dot products.
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],   # id 0
            [0.0, 1.0, 0.0, 0.0],   # id 1
            [0.7071, 0.7071, 0.0, 0.0],  # id 2 -- 45 degrees from both
        ], dtype=np.float32)
        self.ids = _run(self.index.add_batch(vecs))
        self.assertEqual(len(self.ids), 3)

    def test_n1_unchanged(self):
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (D,), single photo
        results = _run(self.index.restricted_search(query, self.ids, top_k=3))
        by_id = {r["faiss_id"]: r for r in results}
        self.assertAlmostEqual(by_id[self.ids[0]]["score"], 1.0, places=4)
        # median_query_idx must always be 0 when there's only one query.
        for r in results:
            self.assertEqual(r["median_query_idx"], 0)

    def test_n3_median_matches_hand_computed(self):
        # Three query photos of "the same animal" as id 0, with one clear
        # outlier (photo index 1) -- exactly the UKDEGR071423 shape from the
        # real harness (two similar photos, one bad one).
        queries = np.array([
            [1.0, 0.0, 0.0, 0.0],    # matches id0 perfectly
            [0.0, 0.0, 1.0, 0.0],    # orthogonal to everything -- the "bad photo"
            [0.9, 0.1, 0.0, 0.0],    # close to id0, slightly off
        ], dtype=np.float32)
        results = _run(self.index.restricted_search(queries, self.ids, top_k=3))
        by_id = {r["faiss_id"]: r for r in results}

        # Against id0: raw scores are [1.0, 0.0, ~0.994]. Median = the middle
        # value = ~0.994 (query index 2), NOT the max (1.0, query 0) and NOT
        # pulled down by the outlier (0.0, query 1).
        r0 = by_id[self.ids[0]]
        self.assertGreater(r0["score"], 0.9)
        self.assertLess(r0["score"], 1.0)  # proves it's not the max
        self.assertEqual(r0["median_query_idx"], 2)

    def test_median_query_idx_is_exact_not_nearest(self):
        """For odd N, the reported median_query_idx's OWN score against that
        candidate must equal the reported median score exactly (float
        precision aside) -- it's an actual sample, not an approximation."""
        queries = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        results = _run(self.index.restricted_search(queries, self.ids, top_k=3))
        candidate_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
        ], dtype=np.float32)
        norm_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        for r in results:
            cand_idx = self.ids.index(r["faiss_id"])
            own_score = float(np.dot(norm_queries[r["median_query_idx"]], candidate_matrix[cand_idx]))
            self.assertAlmostEqual(own_score, r["score"], places=4)


if __name__ == "__main__":
    unittest.main()
