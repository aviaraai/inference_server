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

    def __init__(self, embedding_dim: int = EMB_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
        self._lock = threading.Lock()  # real cross-thread mutual exclusion
        self._model_version: str = MODEL_VERSION
        self._saved_at: str | None = None

    # ── Embedding preparation (pure, no shared state — no lock needed) ──────

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
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

    async def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
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
        return self.index.ntotal
