"""
pipeline/whitening.py — frozen PCA-whitening artifact: load + apply + hash.

The whitening matrix is fit ONCE (scripts/fit_whitening.py) and then frozen.
Refitting it silently changes the embedding space and invalidates every
vector already in the FAISS index, so this loader refuses to run against an
artifact whose content hash doesn't match what it was told to expect (the
index's own metadata should carry the hash it was built with — see
scripts/reindex_gallery.py).

Artifact contract (.npz)
------------------------
mean            (D,)         float32 — fusion-space mean, fit-set-derived
components      (K, D)       float32 — PCA projection rows (top-K, already
                                        divided by sqrt(eigenvalue): "whitened"
                                        rows), so apply() is one matmul
eigenvalues     (K,)         float32 — raw eigenvalues, kept for provenance
                                        and re-derivation, not used at apply time
fit_image_list  (N,) str     — exact image paths the PCA was fit on
fit_rank        ()  int64    — numerical rank of the fit-set covariance;
                                K must be <= this or components past the rank
                                are noise divided by ~zero (see fit_whitening.py)
content_hash    ()  str      — sha256 of (mean, components, eigenvalues) bytes
fit_timestamp   ()  str      — ISO8601 UTC
held_out_eval   ()  str      — JSON string of the validation numbers measured
                                on identities NOT in fit_image_list
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("godhaar.whitening")


def compute_content_hash(mean: np.ndarray, components: np.ndarray, eigenvalues: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mean, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(components, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(eigenvalues, dtype=np.float32).tobytes())
    return h.hexdigest()


@dataclass
class WhiteningTransform:
    mean: np.ndarray            # (D,)
    components: np.ndarray      # (K, D) — pre-divided by sqrt(eigenvalue)
    eigenvalues: np.ndarray     # (K,)
    content_hash: str
    fit_image_list: list[str]
    fit_rank: int
    fit_timestamp: str
    held_out_eval: dict
    n_components: int

    @classmethod
    def load(cls, path: str | Path, *, expected_hash: str | None = None) -> "WhiteningTransform":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Whitening artifact not found: {path}")

        d = np.load(path, allow_pickle=False)
        mean = d["mean"].astype(np.float32)
        components = d["components"].astype(np.float32)
        eigenvalues = d["eigenvalues"].astype(np.float32)
        fit_image_list = [str(s) for s in d["fit_image_list"]]
        fit_rank = int(d["fit_rank"])
        fit_timestamp = str(d["fit_timestamp"])
        content_hash = str(d["content_hash"])
        held_out_eval = json.loads(str(d["held_out_eval"])) if "held_out_eval" in d else {}

        recomputed = compute_content_hash(mean, components, eigenvalues)
        if recomputed != content_hash:
            raise RuntimeError(
                f"Whitening artifact {path} is CORRUPT or was hand-edited: "
                f"stored content_hash={content_hash} != recomputed={recomputed}. "
                f"Refusing to load a whitening matrix that doesn't match its "
                f"own recorded hash."
            )

        if expected_hash is not None and expected_hash != content_hash:
            raise RuntimeError(
                f"Whitening artifact hash mismatch: the FAISS index was built "
                f"with whitening hash={expected_hash!r}, but the artifact at "
                f"{path} has hash={content_hash!r}. Refitting the whitening "
                f"matrix invalidates every stored embedding — either point at "
                f"the artifact the index was actually built with, or "
                f"re-run scripts/reindex_gallery.py against this new artifact "
                f"before serving traffic with it."
            )

        n_components = components.shape[0]
        if n_components > fit_rank:
            raise RuntimeError(
                f"Whitening artifact requests {n_components} components but "
                f"the fit set's numerical rank was only {fit_rank} — "
                f"components past the rank are noise divided by ~zero "
                f"eigenvalues. This artifact should never have been produced; "
                f"refuse to load it rather than serve garbage embeddings."
            )

        log.info(
            f"Whitening loaded: {path} — {n_components} components, "
            f"fit on {len(fit_image_list)} images (rank={fit_rank}), "
            f"hash={content_hash[:12]}..., fit_at={fit_timestamp}"
        )
        return cls(
            mean=mean, components=components, eigenvalues=eigenvalues,
            content_hash=content_hash, fit_image_list=fit_image_list,
            fit_rank=fit_rank, fit_timestamp=fit_timestamp,
            held_out_eval=held_out_eval, n_components=n_components,
        )

    @staticmethod
    def fit(fused: np.ndarray, n_components: int) -> dict:
        """Fit PCA-whitening on a (N, D) L2-normed fusion-embedding matrix.

        Returns a dict of raw arrays (mean, components, eigenvalues,
        content_hash, fit_rank) -- NOT a WhiteningTransform, because the
        caller (scripts/fit_whitening.py) still needs to attach
        fit_image_list/held_out_eval/fit_timestamp before it's a complete,
        loadable artifact. Kept here, not duplicated in the fit script or
        the validation gate, so both use the exact same math.

        Refuses (raises) if n_components exceeds the fit set's own
        numerical rank -- see scripts/fit_whitening.py's module docstring
        for why (512/1024 components collapsed to 16.9% top-1 in the
        reference sweep when the fit set only supported ~312).
        """
        fused = np.asarray(fused, dtype=np.float64)
        mean = fused.mean(axis=0)
        centered = fused - mean[None, :]
        # SVD of the centered fit matrix: singular values S relate to PCA
        # eigenvalues of the covariance by eigval = S^2 / (n-1).
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        n = centered.shape[0]
        eigenvalues_full = (s ** 2) / max(1, n - 1)
        fit_rank = int(np.sum(s > s.max() * 1e-10)) if s.size else 0

        if n_components > fit_rank:
            raise ValueError(
                f"Requested n_components={n_components} exceeds the fit "
                f"set's numerical rank={fit_rank} (n_samples={n}). "
                f"Components past the rank divide by a ~zero eigenvalue and "
                f"produce noise, not signal -- reduce n_components or fit on "
                f"more images."
            )

        components_raw = vt[:n_components]              # (K, D)
        eigenvalues = eigenvalues_full[:n_components]    # (K,)
        components = components_raw / (np.sqrt(eigenvalues)[:, None] + 1e-6)

        mean32 = mean.astype(np.float32)
        components32 = components.astype(np.float32)
        eigenvalues32 = eigenvalues.astype(np.float32)
        content_hash = compute_content_hash(mean32, components32, eigenvalues32)

        return {
            "mean": mean32,
            "components": components32,
            "eigenvalues": eigenvalues32,
            "fit_rank": fit_rank,
            "content_hash": content_hash,
        }

    def apply(self, fused: np.ndarray) -> np.ndarray:
        """fused: (B, D) L2-normed fusion vectors -> (B, K) L2-normed whitened."""
        centered = fused - self.mean[None, :]
        projected = centered @ self.components.T  # (B, K), already /sqrt(eigval)
        norm = np.linalg.norm(projected, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        return projected / norm
