"""
faiss_index.py — Thread-safe, persistent, versioned FAISS index.

Changes from original:
  1. asyncio.Lock() around all write operations (add, remove).
  2. save/load persistence to disk.
  3. Model version metadata stored alongside the index.
  4. cattle_search() returns results with cattle_id via IDStore lookup.
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from godhaar.config import EMB_DIM, MODEL_VERSION
from id_store import IDStore

log = logging.getLogger("godhaar.faiss_index")


class FaissIndex:
    """Thread-safe FAISS IndexIDMap2 with persistence and versioning.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of embeddings (256 for GodhaarModel).
    """

    def __init__(self, embedding_dim: int = EMB_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
        self._write_lock = asyncio.Lock()
        self._model_version: str = MODEL_VERSION
        self._created_at: Optional[str] = None

    # ── Embedding preparation ────────────────────────────────────────────────

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1D.")

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}, got {embedding.shape[0]}")

        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)

        return embedding

    def _generate_id(self) -> np.int64:
        """Generate a random positive int64."""
        return np.int64(secrets.randbits(63))

    # ── Write operations (locked) ────────────────────────────────────────────

    async def add(self, embedding: np.ndarray) -> int:
        """Add a single embedding to the index. Thread-safe.

        Parameters
        ----------
        embedding : np.ndarray, shape (256,)

        Returns
        -------
        int : the generated FAISS ID.
        """
        vector = self._prepare_embedding(embedding)
        item_id = self._generate_id()

        async with self._write_lock:
            self.index.add_with_ids(
                vector,
                np.array([item_id], dtype=np.int64),
            )

        return int(item_id)

    async def add_batch(self, embeddings: np.ndarray) -> list[int]:
        """Add a batch of embeddings. Thread-safe.

        Parameters
        ----------
        embeddings : np.ndarray, shape (B, 256)

        Returns
        -------
        list[int] : generated FAISS IDs.
        """
        ids = []
        vectors = []
        for i in range(embeddings.shape[0]):
            vec = self._prepare_embedding(embeddings[i])
            fid = self._generate_id()
            vectors.append(vec)
            ids.append(int(fid))

        if vectors:
            all_vecs = np.vstack(vectors)
            all_ids = np.array(ids, dtype=np.int64)

            async with self._write_lock:
                self.index.add_with_ids(all_vecs, all_ids)

        return ids

    # ── Read operations (no lock needed for FAISS reads) ─────────────────────

    def search(self, embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Search for the nearest neighbours of an embedding.

        Parameters
        ----------
        embedding : np.ndarray, shape (256,)
        top_k : int

        Returns
        -------
        list of {"faiss_id": int, "score": float}
        """
        if self.index.ntotal == 0:
            return []

        vector = self._prepare_embedding(embedding)

        scores, ids = self.index.search(
            vector,
            min(top_k, self.index.ntotal),
        )

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append({
                "faiss_id": int(idx),
                "score": float(score),
            })

        return results

    def cattle_search(
        self,
        embedding: np.ndarray,
        id_store: IDStore,
        top_k: int = 10,
    ) -> list[dict]:
        """Search FAISS and resolve results to cattle_ids.

        Also computes the gap between consecutive matches.

        Parameters
        ----------
        embedding : np.ndarray, shape (256,)
        id_store : IDStore
        top_k : int

        Returns
        -------
        list of {"rank": int, "cattle_id": str, "score": float, "gap": float | None}
        """
        raw_results = self.search(embedding, top_k=top_k * 3)  # overfetch for dedup

        if not raw_results:
            return []

        # Resolve FAISS IDs to cattle_ids
        faiss_ids = [r["faiss_id"] for r in raw_results]
        id_map = id_store.lookup_batch(faiss_ids)

        # Aggregate by cattle_id (keep max score per cattle)
        best: dict[str, float] = {}
        for r in raw_results:
            cattle_id = id_map.get(r["faiss_id"])
            if cattle_id is None:
                continue
            if cattle_id not in best or r["score"] > best[cattle_id]:
                best[cattle_id] = r["score"]

        # Sort by score descending
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build results with gap
        results = []
        for i, (cattle_id, score) in enumerate(ranked):
            gap = round(score - ranked[i + 1][1], 6) if i + 1 < len(ranked) else None
            results.append({
                "rank": i + 1,
                "cattle_id": cattle_id,
                "score": round(score, 6),
                "gap": gap,
            })

        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    async def save(self, index_path: str | Path, meta_path: Optional[str | Path] = None) -> None:
        """Save the FAISS index and metadata to disk. Thread-safe.

        Parameters
        ----------
        index_path : path to write the .index file
        meta_path : path to write the .meta.json file (defaults to same dir)
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if meta_path is None:
            meta_path = index_path.with_suffix(".meta.json")
        meta_path = Path(meta_path)

        async with self._write_lock:
            faiss.write_index(self.index, str(index_path))

        meta = {
            "embedding_dim": self.embedding_dim,
            "model_version": self._model_version,
            "total_vectors": self.index.ntotal,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info(f"FAISS index saved: {index_path} ({self.index.ntotal} vectors, model={self._model_version})")

    def load(self, index_path: str | Path, meta_path: Optional[str | Path] = None) -> None:
        """Load a FAISS index and validate model version compatibility.

        Parameters
        ----------
        index_path : path to the .index file
        meta_path : path to the .meta.json file
        """
        index_path = Path(index_path)
        if not index_path.exists():
            log.warning(f"FAISS index not found at {index_path}. Starting with empty index.")
            return

        if meta_path is None:
            meta_path = index_path.with_suffix(".meta.json")
        meta_path = Path(meta_path)

        # Load and validate metadata
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            stored_version = meta.get("model_version", "unknown")
            if stored_version != self._model_version:
                log.warning(
                    f"⚠ FAISS index was built with model '{stored_version}' but current model "
                    f"is '{self._model_version}'. Embeddings may be incompatible!"
                )
            self._created_at = meta.get("saved_at")
            log.info(f"FAISS meta: model={stored_version}, saved_at={self._created_at}")

        self.index = faiss.read_index(str(index_path))
        log.info(f"FAISS index loaded: {index_path} ({self.index.ntotal} vectors)")

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def faiss_version_label(self) -> str:
        """Return a date-based label for the index version."""
        if self._created_at:
            return self._created_at[:10]
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def __len__(self) -> int:
        return self.index.ntotal
