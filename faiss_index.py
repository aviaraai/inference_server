import secrets

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))

    def _prepare_embedding(self, embedding: np.ndarray):
        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1D.")

        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}, got {embedding.shape[0]}")

        embedding = embedding.reshape(1, -1)

        faiss.normalize_L2(embedding)

        return embedding

    def _generate_id(self) -> np.int64:
        """
        Generate a random positive int64.
        """
        return np.int64(secrets.randbits(63))

    def add(self, embedding: np.ndarray) -> int:
        vector = self._prepare_embedding(embedding)

        item_id = self._generate_id()

        self.index.add_with_ids(
            vector,
            np.array([item_id], dtype=np.int64),
        )

        return int(item_id)

    def search(self, embedding: np.ndarray, top_k: int = 5):
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

            results.append(
                {
                    "id": int(idx),
                    "score": float(score),
                }
            )

        return results

    def __len__(self):
        return self.index.ntotal
