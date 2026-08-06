"""
utils.py — Helper functions for color space conversion, statistics, and dominant-color clustering.
"""

import cv2
import numpy as np

_EMPTY_FEATURES = {
    "dominant_lab": [0.0, 0.0, 0.0],
    "confidence": 0.0,
    "peak_ratio": 0.0,
    "is_multimodal": False,
    "clusters": [],
}

# Deterministic k-means config. classify_body_color()'s caller (register)
# requires two front photos of the same animal to agree on a label, so the
# same image must always cluster to the same result — hence the fixed seed
# rather than OpenCV/numpy's default (time-seeded) RNG.
_CLUSTER_K = 4
_CLUSTER_SEED = 42
_MAX_CLUSTER_SAMPLE = 8000  # cap pixels fed to k-means; a full-res ROI can be millions

# Spatial prior used ONLY when no foreground mask is available (see
# _center_weights). Sigma is in units of "fraction of the way from ROI
# center to its edge": 0.60 gives weight 1.0 dead center, ~0.25 at the
# edge midpoints, ~0.06 in the corners.
_CENTER_PRIOR_SIGMA = 0.60


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to CIE L*a*b*."""
    if img_bgr is None or img_bgr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)


def parse_raw_lab(pixel_lab: np.ndarray | list | tuple) -> list[float]:
    """Convert OpenCV representation of LAB [0-255] back to standard CIE LAB scale:

        L* in [0, 100]
        a* in [-128, 127]
        b* in [-128, 127]
    """
    l_raw, a_raw, b_raw = pixel_lab[0], pixel_lab[1], pixel_lab[2]
    l_std = float(l_raw) * 100.0 / 255.0
    a_std = float(a_raw) - 128.0
    b_std = float(b_raw) - 128.0
    return [round(l_std, 2), round(a_std, 2), round(b_std, 2)]


def calculate_median_lab(img_lab: np.ndarray) -> list[float]:
    """Calculate the median L*, a*, b* values of the image on standard CIE scale."""
    if img_lab is None or img_lab.size == 0:
        return [0.0, 0.0, 0.0]

    median_raw = np.median(img_lab.reshape(-1, 3), axis=0)
    return parse_raw_lab(median_raw)


def _kmeans_lab(pixels: np.ndarray, k: int, seed: int, max_iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means over full L*a*b* pixels (fixed seed -> fixed output for a fixed input).

    k-means++ initialization for stable convergence with few clusters, then
    standard Lloyd's-algorithm refinement. Pure numpy (no sklearn dependency
    in this repo) — the pixel counts here (<= _MAX_CLUSTER_SAMPLE) keep the
    O(n*k) distance matrix small.
    """
    n = len(pixels)
    k = min(k, n)
    rng = np.random.RandomState(seed)

    centers = np.empty((k, 3), dtype=np.float64)
    first = rng.randint(n)
    centers[0] = pixels[first]
    closest_sq_dist = ((pixels - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest_sq_dist.sum()
        if total > 0:
            next_idx = rng.choice(n, p=closest_sq_dist / total)
        else:
            # All remaining points coincide with already-chosen centers.
            next_idx = rng.randint(n)
        centers[i] = pixels[next_idx]
        new_sq_dist = ((pixels - centers[i]) ** 2).sum(axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    labels = np.full(n, -1, dtype=np.int32)
    for _ in range(max_iters):
        dists = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                centers[i] = cluster_pixels.mean(axis=0)
            # Empty cluster: leave its center in place. It simply won't be
            # picked as the largest/second-largest cluster below.

    return centers, labels


def aggregate_clusters_by_label(clusters: list[dict], classify_fn, centrality_ratio_min: float = 0.0) -> list[tuple[str, float]]:
    """Group color clusters by their classified label and sum their weights.

    Clustering runs over L* as well as a*/b*, so one perceptual coat color
    routinely fragments across several clusters that differ only in lighting
    — a sunlit flank and a shadowed flank of the same brown cow come back as
    two separate BROWN clusters. Reading the color off the single heaviest
    CLUSTER therefore understates the true dominant color and can hand the
    answer to a smaller, genuinely different-colored region: on a real Gir
    photo the coat split into BROWN 0.29 + BROWN 0.27 while a white blaze
    formed one 0.29 cluster, so "heaviest cluster" was a coin-flip that the
    coat could lose. Summing by LABEL first asks the question that actually
    matters — how much of this animal is brown vs white.

    Parameters
    ----------
    classify_fn : callable
        Maps a [L*, a*, b*] centroid to a label string.
    centrality_ratio_min : float
        Drop clusters less than this fraction as central as the most central
        cluster. Used on un-localized crops to discard the surrounding scene,
        which is a real, differently-colored region but is not the animal.
        Pass 0.0 to disable (e.g. when a foreground mask already did this).

    Returns
    -------
    list of (label, summed_weight), heaviest first. Empty if no clusters.
    """
    if not clusters:
        return []

    max_centrality = max(c.get("centrality", 1.0) for c in clusters)
    totals: dict[str, float] = {}
    for c in clusters:
        if centrality_ratio_min > 0.0 and c.get("centrality", 1.0) < centrality_ratio_min * max_centrality:
            continue
        label = classify_fn(c["lab"])
        totals[label] = totals.get(label, 0.0) + c["weight"]

    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def _center_weights(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Soft spatial prior favouring pixels near the ROI center.

    Used ONLY when no foreground mask is available. Selecting the dominant
    color by raw pixel count is wrong on an un-localized crop: if the animal
    occupies a minority of the frame (routine in real field photos), the
    single biggest cluster is the BACKGROUND, so the classifier confidently
    reports the color of the dirt behind the cow.

    The fixed center crop this falls back to already assumes "the subject is
    in the middle of the frame" — that is the only reason cropping a fixed
    70%x60% box is a sane thing to do at all. This applies the same
    assumption more honestly: instead of a hard binary in/out crop, weight
    each pixel's vote by how central it is, so peripheral background can
    still be outvoted by a smaller but centered subject.

    Returns a weight per pixel in (0, 1].
    """
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Normalized elliptical distance: 0 at center, 1.0 at edge midpoints.
    dy = (coords[:, 0] - cy) / max(cy, 1e-6)
    dx = (coords[:, 1] - cx) / max(cx, 1e-6)
    d_sq = dy ** 2 + dx ** 2
    return np.exp(-d_sq / (2.0 * _CENTER_PRIOR_SIGMA ** 2))


def extract_dominant_lab_features(img_lab: np.ndarray, k: int = _CLUSTER_K) -> dict:
    """Find the dominant coat color via k-means clustering in full L*a*b* space.

    Replaces a prior 2D a*/b*-histogram approach that had two bugs: it
    returned the HISTOGRAM BIN CENTER as "dominant_lab" instead of a real
    pixel value (the neutral point sat on a bin edge, so any near-neutral
    grey/black/white coat always landed in the same bin and reported an
    identical fabricated chroma), and it clustered only on a*/b*, ignoring
    L* entirely — so a black coat and a white coat (both near-zero chroma)
    fell into the same bin and had their lightness values merged, which is
    what actually decided BLACK vs WHITE.

    Clustering all three channels together keeps black/white/grey/brown
    coats in separate clusters, and returns each cluster's real centroid
    rather than a bin midpoint.

    Returns
    -------
    dict containing:
        dominant_lab : list[float]  # [L*, a*, b*] centroid of the dominant cluster
        confidence : float         # dominant cluster's share of total pixel weight
        peak_ratio : float         # 2nd-heaviest cluster weight / dominant cluster weight
        is_multimodal : bool       # True if there's a substantial secondary cluster
        clusters : list[dict]      # every cluster as {"lab": [...], "weight": float},
                                   # heaviest first. Callers needing to know whether a
                                   # secondary cluster is a genuinely DIFFERENT COLOR
                                   # (spotted/mixed coats) must classify these centroids
                                   # rather than trusting peak_ratio alone — with
                                   # clustering over L* too, a solid coat splits into
                                   # several lit/shadowed clusters of the SAME color.
    """
    if img_lab is None or img_lab.size == 0:
        return dict(_EMPTY_FEATURES)

    h, w = img_lab.shape[:2]
    ys, xs = np.divmod(np.arange(h * w), w)
    pixels = img_lab.reshape(-1, 3)

    total_pixels = len(pixels)
    if total_pixels == 0:
        return dict(_EMPTY_FEATURES)

    pixels = pixels.astype(np.float64)
    coords = np.column_stack((ys, xs))

    rng = np.random.RandomState(_CLUSTER_SEED)
    if total_pixels > _MAX_CLUSTER_SAMPLE:
        sample_idx = rng.choice(total_pixels, _MAX_CLUSTER_SAMPLE, replace=False)
        sample, sample_coords = pixels[sample_idx], coords[sample_idx]
    else:
        sample, sample_coords = pixels, coords

    centers, labels = _kmeans_lab(sample, k, seed=_CLUSTER_SEED)

    weights = _center_weights(sample_coords, (h, w))

    cluster_weights = np.bincount(labels, weights=weights, minlength=len(centers))
    total_weight = float(cluster_weights.sum())
    order = np.argsort(cluster_weights)[::-1]

    primary_weight = float(cluster_weights[order[0]])
    secondary_weight = float(cluster_weights[order[1]]) if len(order) > 1 else 0.0

    # Per-cluster centrality: how central, on average, its pixels are. A
    # secondary cluster that is a different color AND sits on the subject is
    # a spot/patch; one that hugs the frame edges is background.
    clusters = []
    for i in order:
        if cluster_weights[i] <= 0:
            continue
        member = labels == i
        centrality = float(weights[member].mean())
        clusters.append({
            "lab": parse_raw_lab(centers[i]),
            "weight": round(float(cluster_weights[i] / total_weight), 4),
            "centrality": round(centrality, 4),
        })

    peak_ratio = float(secondary_weight / primary_weight) if primary_weight > 0 else 0.0
    confidence = float(primary_weight / total_weight) if total_weight > 0 else 0.0

    return {
        "dominant_lab": parse_raw_lab(centers[order[0]]),
        "confidence": round(confidence, 4),
        "peak_ratio": round(peak_ratio, 4),
        "is_multimodal": peak_ratio >= 0.25,
        "clusters": clusters,
    }
