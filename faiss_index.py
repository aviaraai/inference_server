import asyncio
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from godhaar.config import EMB_DIM, MODEL_VERSION

log = logging.getLogger("godhaar.faiss_index")


class FaissIndex:
    """Thread-safe FAISS IndexIDMap2 with persistence and versioning.

    Locking model
    -------------
    All FAISS index access (reads AND writes) is guarded by a single
    threading.Lock, not an asyncio.Lock. The index is a native C++ object
    shared across OS threads (since blocking calls are offloaded via
    asyncio.to_thread), so the thing protecting it must provide real
    mutual exclusion between threads — independent of the GIL, and safe
    under a future no-GIL Python. asyncio.Lock only serializes coroutines
    on one event loop; it does nothing for cross-thread access, so it is
    never used for the index itself.

    All public methods are async and offload blocking FAISS/numpy work
    to a thread via asyncio.to_thread, so the event loop is never blocked.
    """

    # Maximum number of per-ID reconstruct() calls permitted when the native
    # reconstruct_batch() is unavailable (old FAISS build). Above this limit
    # we raise rather than silently issue hundreds/thousands of C++ calls.
    _RECONSTRUCT_FALLBACK_LIMIT: int = 64

    def __init__(self, embedding_dim: int = EMB_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
        self._lock = threading.Lock()  # real cross-thread mutual exclusion
        self._model_version: str = MODEL_VERSION
        self._saved_at: str | None = None

    # ── Embedding preparation (pure, no shared state — no lock needed) ──────

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cast, reshape, validate, and L2-normalize an embedding array.

        This is the single normalization point for ALL vectors entering the
        index. Vectors stored via add_batch() are normalized here and remain
        unit-norm permanently — they are never re-normalized on read.
        """
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be shape (D,) or (N, D)")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected dimension {self.embedding_dim}, got {vectors.shape[1]}"
            )
        vectors = vectors.copy()
        faiss.normalize_L2(vectors)
        return vectors

    async def add_batch(
        self,
        embeddings: np.ndarray,
    ) -> list[int]:

        vectors = self._prepare_embeddings(embeddings)

        if vectors.shape[0] == 0:
            return []

        ids = np.array(
            [secrets.randbits(63) for _ in range(vectors.shape[0])],
            dtype=np.int64,
        )

        await asyncio.to_thread(
            self._add_batch_sync,
            vectors,
            ids,
        )

        return ids.tolist()

    def _add_batch_sync(self, all_vecs: np.ndarray, all_ids: np.ndarray) -> None:
        with self._lock:
            self.index.add_with_ids(all_vecs, all_ids)

    # ── Read operations ──────────────────────────────────────────────────────
    # NOTE: reads now take the same lock as writes. FAISS gives no guarantee
    # that a read is safe concurrently with a write on the same index, so
    # "reads need no lock" (the old comment) is incorrect once writes can run
    # on a separate thread. A single Lock trades a little read/read
    # concurrency for correctness; if profiling shows read contention is a
    # bottleneck, upgrade to a reader-writer lock rather than dropping this.

    async def reconstruct_batch(
        self,
        ids: list[int],
    ) -> np.ndarray:
        """Reconstruct stored vectors for the given FAISS IDs.

        Returns an (N, D) float32 matrix. Uses FAISS native
        ``reconstruct_batch`` when available, otherwise falls back to
        per-ID ``reconstruct``.
        """
        return await asyncio.to_thread(self._reconstruct_batch_sync, ids)

    def _reconstruct_batch_sync(self, ids: list[int]) -> np.ndarray:
        """Lock-holding entry point — used by the public async method."""
        with self._lock:
            return self._reconstruct_batch_inner(ids)

    def _reconstruct_batch_inner(self, ids: list[int]) -> np.ndarray:
        """Lock-free core — caller MUST already hold self._lock.

        Strategy
        --------
        1. Use FAISS native ``reconstruct_batch`` when available — one C++ call,
           fast for any N.
        2. If the method doesn't exist (old FAISS build), fall back to per-ID
           ``reconstruct`` only when N is small (≤ _RECONSTRUCT_FALLBACK_LIMIT).
        3. If native batch raises at runtime, or N exceeds the fallback limit,
           re-raise immediately rather than silently degrading to thousands of
           individual calls.
        """
        id_arr = np.array(ids, dtype=np.int64)

        if hasattr(self.index, "reconstruct_batch"):
            # Native path — let any exception propagate; do NOT catch & loop.
            return self.index.reconstruct_batch(id_arr)

        # reconstruct_batch unavailable (old FAISS build).
        # Only tolerate per-ID loop for small batches.
        if len(ids) > self._RECONSTRUCT_FALLBACK_LIMIT:
            raise RuntimeError(
                f"FAISS native reconstruct_batch is unavailable and the candidate "
                f"batch ({len(ids)} ids) exceeds the per-ID fallback limit "
                f"({self._RECONSTRUCT_FALLBACK_LIMIT}). Upgrade faiss-cpu/faiss-gpu."
            )

        log.warning(
            "FAISS reconstruct_batch unavailable — using per-ID fallback "
            f"for {len(ids)} candidates. Upgrade faiss to remove this path."
        )
        return np.vstack([self.index.reconstruct(int(i)) for i in ids])

    async def restricted_search(
        self,
        query_embedding: np.ndarray,
        candidate_ids: list[int],
        top_k: int = 5,
    ) -> list[dict]:
        """Rank only the given candidates against one or more query
        embeddings.

        Normalization contract
        ----------------------
        Candidate vectors are already unit-norm — they were normalized once at
        registration via _prepare_embeddings() and stored that way permanently.
        Only the query is normalized here. Re-normalizing candidates on every
        search would be wasted CPU.

        Multi-photo query (query_embedding shape (N, D), N > 1)
        ---------------------------------------------------------
        Per-candidate scores are the MEDIAN across the N query embeddings,
        not the max. Verified offline (scratch_multiphoto_search_eval.py,
        leave-one-out over 189 real animals, N=3): plain max-aggregation
        widens the true-positive score but ALSO inflates the best impostor's
        score via the same multiple-comparisons effect (each impostor gets a
        full N-vs-3 max, not a fair single comparison), which measurably
        REDUCED real-MATCH recall (21.5% -> 16.4%). Median rescued 84/567
        single-photo failures against only 5 regressions, and WIDENED the
        gap on average (0.062 -> 0.078) instead of narrowing it. See
        CLAUDE.md's multi-photo search investigation for the full numbers.

        median_query_idx (only meaningful for N > 1)
        ----------------------------------------------
        For each returned candidate, also records which query embedding's
        individual score against it equals the reported median. For an ODD
        N this is exact, not a heuristic: the median of an odd-sized set is
        always one of its actual members (the middle-ranked one after
        sorting), never an interpolation. This is what lets the /search
        route hand LightGlue ONE specific query crop — the one that
        actually produced the reported score — instead of running LightGlue
        N times or picking an arbitrary photo. (For an even N this would be
        an approximation — nearest sample to the interpolated median — but
        this service only ever sends N=3.)

        Flow: reconstruct candidates (unit-norm) → normalize query/queries →
        queries @ candidates.T → per-candidate median across queries → sort
        → return top-k. No FAISS search() call is made.
        """
        return await asyncio.to_thread(
            self._restricted_search_sync,
            query_embedding,
            candidate_ids,
            top_k,
        )

    def _restricted_search_sync(
        self,
        query_embedding: np.ndarray,
        candidate_ids: list[int],
        top_k: int,
    ) -> list[dict]:
        if not candidate_ids:
            return []

        # Normalize query/queries. _prepare_embeddings already accepts (D,)
        # or (N, D) -- reshapes/validates either way, nothing to add here.
        query = self._prepare_embeddings(query_embedding)  # (N, D), N>=1

        with self._lock:
            # Reconstruct candidate vectors — already unit-norm from registration.
            candidate_matrix = self._reconstruct_batch_inner(candidate_ids)  # (C, D)

        # Both sides are unit-norm, so dot product == cosine similarity.
        scores_matrix = np.dot(query, candidate_matrix.T)  # (N, C)

        # Per-candidate median across the N query embeddings. For N=1 this
        # is just that one score, unchanged from before -- median of a
        # single value is itself.
        scores = np.median(scores_matrix, axis=0)  # (C,)

        # Which query embedding produced each candidate's median score --
        # see median_query_idx in the docstring above. argmin over |score -
        # median| is the general form; for odd N one entry is an exact
        # zero-distance match, not merely the closest.
        median_query_idx = np.argmin(np.abs(scores_matrix - scores[np.newaxis, :]), axis=0)  # (C,)

        # Sort descending, take top-k
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        sorted_scores = scores[top_indices]

        results = []
        for idx, i in enumerate(top_indices):
            current = sorted_scores[idx]
            next_score = (
                sorted_scores[idx + 1]
                if idx + 1 < len(sorted_scores)
                else current
            )
            results.append({
                "faiss_id": int(candidate_ids[i]),
                "score": float(current),
                "rank": idx + 1,
                "gap": float(current - next_score),
                "median_query_idx": int(median_query_idx[i]),
            })
        return results

    async def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Full index-wide FAISS search. Used only when no candidate list is
        provided (e.g. internal tooling, health checks). The /search endpoint
        uses restricted_search() instead."""
        return await asyncio.to_thread(
            self._search_sync,
            embedding,
            top_k,
        )

    def _search_sync(
        self,
        embedding: np.ndarray,
        top_k: int,
    ) -> list[dict]:

        vector = self._prepare_embeddings(embedding)

        with self._lock:
            if self.index.ntotal == 0:
                return []

            scores, ids = self.index.search(
                vector,
                min(top_k, self.index.ntotal),
            )

        return [
            {
                "faiss_id": int(idx),
                "score": float(score),
            }
            for score, idx in zip(scores[0], ids[0])
            if idx != -1
        ]

    # ── Persistence ──────────────────────────────────────────────────────────

    async def save(
        self, index_path: str | Path, meta_path: Optional[str | Path] = None
    ) -> None:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_path is None:
            meta_path = index_path.with_suffix(".meta.json")
        meta_path = Path(meta_path)

        ntotal = await asyncio.to_thread(self._save_sync, index_path)

        meta = {
            "embedding_dim": self.embedding_dim,
            "model_version": self._model_version,
            "total_vectors": ntotal,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        log.info(
            f"FAISS index saved: {index_path} ({ntotal} vectors, model={self._model_version})"
        )

    def _save_sync(self, index_path: Path) -> int:
        with self._lock:
            faiss.write_index(self.index, str(index_path))
            return self.index.ntotal

    def load(
        self, index_path: str | Path, meta_path: Optional[str | Path] = None
    ) -> None:
        """Called once during startup before the server accepts traffic —
        no concurrent access is possible yet, so no lock is needed here."""
        index_path = Path(index_path)
        if not index_path.exists():
            log.warning(
                f"FAISS index not found at {index_path}. Starting with empty index."
            )
            return

        if meta_path is None:
            meta_path = index_path.with_suffix(".meta.json")
        meta_path = Path(meta_path)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            stored_version = meta.get("model_version", "unknown")
            if stored_version != self._model_version:
                log.warning(
                    f"⚠ FAISS index was built with model '{stored_version}' but current model "
                    f"is '{self._model_version}'. Embeddings may be incompatible!"
                )
            self._saved_at = meta.get("saved_at")
            log.info(f"FAISS meta: model={stored_version}, saved_at={self._saved_at}")

        self.index = faiss.read_index(str(index_path))
        log.info(f"FAISS index loaded: {index_path} ({self.index.ntotal} vectors)")

    # Additional info

    @property
    def faiss_version_label(self) -> str:
        if self._saved_at:
            return self._saved_at[:10]
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def __len__(self) -> int:
        with self._lock:
            return self.index.ntotal
