"""
scratch_multiphoto_search_eval.py — offline leave-one-out harness for the
multi-photo /search scoping task. Answers ONE question with real data before
any production code gets written: does sending 2-3 query muzzle photos
(instead of 1) actually improve match recall, and does it cost anything in
false-positive risk?

Uses the real photo set at D:\\Group Projects\\godhaar-all-images\\godhaar\\ --
224 UK animals, 7 files each (front1/front2/left1/muzzle1/muzzle2/muzzle3/
right1). Only the 3 muzzle photos per animal are used here.

⚠️ BLOCKED ON REAL WEIGHTS AS OF THIS WRITING: the actual trained embedding
checkpoint (MODEL_PATH / best_top1.pt) is not present anywhere on this
machine (checked: appstorage/Models/ only has pose_model/, no
embedding_model/; the Godhaar/ directory CLAUDE.md references no longer
exists at that path; no .pt/.pth over 50MB anywhere under D:\\Group Projects).
Run with --model-path pointing at the real checkpoint once it's available
(on whatever host actually has it). --synthetic-selftest below is NOT a
substitute for that -- it only proves this script's OWN logic (leave-one-out
gallery construction, per-round max, N=3 aggregation, metrics) is correct on
a known-answer synthetic case; it produces no evidence about real accuracy
and must never be reported as if it did.

── Design (leave-one-out, resolving the "only 3 photos per animal" tension) ──

Naive 3-photo-per-animal leave-one-out breaks down for an N=3 query test:
holding out all 3 photos as queries leaves nothing to index for the true
match. The fix used here: run 3 independent leave-one-out ROUNDS per animal
(one held-out photo each), but ONLY the queried animal's own gallery shrinks
in a given round -- every other (impostor) animal keeps its full 3-photo
gallery, always. Shrinking impostor galleries too would understate real
cross-animal similarity and make false-positive risk look artificially safer
than production.

  Round i, animal A's query = photo i of A.
    A's own candidate score  = max over A's OTHER 2 photos (never itself).
    Impostor B's score       = max over ALL 3 of B's photos (unchanged).

  N=1 condition: pool all 3 rounds x 224 animals = 672 independent trials.
  N=3 condition: per animal (224 of them), take the MAX across its 3 rounds'
    per-candidate scores, then re-rank. Because impostor galleries are
    identical across rounds, this collapses to exactly "A's 3 query photos x
    B's 3 stored photos, overall max" -- the real quantity multi-photo query
    would compute in production. For the true-positive score it becomes the
    best pairwise match among A's own 3 photos, excluding self-matches.

Metrics (defined here, not assumed):
  strict match rate = top-1 accuracy (true animal is the single best score).
  robust recall      = fraction reaching decide()'s real MATCH verdict:
                        score >= 0.82 (matchThreshold) AND
                        gap   >= 0.08 (gapThreshold) -- decision.go's exact
                        constants, no attribute adjustment (no color/horn
                        data being tested here).
  score / gap distributions for the true-positive case, both conditions.

Usage:
    # Real run, once weights exist somewhere reachable:
    .venv/Scripts/python.exe scratch_multiphoto_search_eval.py \
        --model-path /path/to/best_top1.pt --yolo-path yolov8s.pt

    # Logic self-test only (no real model needed, proves nothing about accuracy):
    .venv/Scripts/python.exe scratch_multiphoto_search_eval.py --synthetic-selftest
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "godhaar-all-images", "godhaar"
)

MATCH_THRESHOLD = 0.82  # decision.go's matchThreshold, verbatim
GAP_THRESHOLD = 0.08    # decision.go's gapThreshold, verbatim


# ── Data loading ──────────────────────────────────────────────────────────────

def find_uk_muzzle_photos(dataset_dir: str) -> dict[str, list[str]]:
    """{tag: [path_muzzle1, path_muzzle2, path_muzzle3]} for every UKDE*
    animal with all 3 muzzle photos present. Reports (doesn't silently drop)
    any tag missing a slot."""
    pattern = os.path.join(dataset_dir, "UKDE*_muzzle*.jpg")
    paths = sorted(glob.glob(pattern))
    by_tag: dict[str, dict[int, str]] = defaultdict(dict)
    for p in paths:
        m = re.match(r"^(UKDE\w+)_muzzle(\d)\.jpg$", os.path.basename(p))
        if not m:
            continue
        by_tag[m.group(1)][int(m.group(2))] = p

    complete: dict[str, list[str]] = {}
    incomplete: list[str] = []
    for tag, slots in by_tag.items():
        if set(slots.keys()) == {1, 2, 3}:
            complete[tag] = [slots[1], slots[2], slots[3]]
        else:
            incomplete.append(f"{tag} (has slots {sorted(slots.keys())})")

    if incomplete:
        print(f"WARNING: {len(incomplete)} tags missing a muzzle slot, excluded: {incomplete[:10]}"
              + (" ..." if len(incomplete) > 10 else ""))
    return complete


# ── Embedding backends ────────────────────────────────────────────────────────

class RealEmbedder:
    """The actual production path: crop_cattle -> GodhaarModel. Requires real
    weights -- this is what produces numbers worth reporting."""

    def __init__(self, model_path: str, yolo_path: str):
        from godhaar.model import GodhaarModel
        from pipeline.muzzle import embed_batch
        from pipeline.yolo_crop import crop_cattle, load_yolo

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ckpt = GodhaarModel.load_checkpoint(model_path, device=self.device)
        self.model.eval()
        load_yolo(yolo_path)
        self._crop_cattle = crop_cattle
        self._embed_batch = embed_batch

    def detect_crop(self, path: str) -> tuple[np.ndarray | None, str]:
        """Runs crop_cattle on ONE photo. Returns (crop_or_None, status) --
        status is always reported, even on success, so the caller can build
        an exact per-photo failure count across the whole dataset rather than
        only knowing "this animal had a problem somewhere"."""
        img = cv2.imread(path)
        if img is None:
            return None, "UNREADABLE"
        crop, det_status, _conf = self._crop_cattle(img)
        return crop, det_status

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Already-detected crops -> (len(crops), 256) unit-norm embeddings,
        batched in one forward pass -- same as /register's own embed_batch
        call on its 3 muzzle crops."""
        jpg_bytes = [cv2.imencode(".jpg", c)[1].tobytes() for c in crops]
        emb = self._embed_batch(jpg_bytes, self.model, self.device)
        return emb.numpy()

    @staticmethod
    def quality_ok(crop: np.ndarray) -> tuple[bool, str]:
        """The real production quality gate (pipeline/quality.py's
        quality_check_cv2 -- blur/exposure/size, BLUR_THRESHOLD=20.0 Laplacian
        variance, godhaar/config.py), run on the already-detected crop. This
        is exactly what /register and /search already call today -- not a
        new threshold invented for this harness."""
        from pipeline.quality import quality_check_cv2

        status, reason = quality_check_cv2(crop)
        return status == "GOOD", reason


class SyntheticEmbedder:
    """Logic self-test ONLY. Each animal gets one random base vector; its 3
    'embeddings' are that base plus small noise, unit-normalized -- high
    self-similarity, ~orthogonal to other animals' bases, by construction.
    Proves the harness's OWN math is correct; proves NOTHING about real
    accuracy. Never report these numbers as evaluation results."""

    def __init__(self, dim: int = 256, noise: float = 0.03, seed: int = 0):
        self.dim = dim
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self._bases: dict[str, np.ndarray] = {}

    def embed_photos(self, paths: list[str], tag: str | None = None) -> np.ndarray:
        if tag not in self._bases:
            v = self.rng.normal(size=self.dim)
            self._bases[tag] = v / np.linalg.norm(v)
        base = self._bases[tag]
        out = []
        for _ in paths:
            v = base + self.rng.normal(scale=self.noise, size=self.dim)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


# ── Leave-one-out evaluation ──────────────────────────────────────────────────
#
# The expensive part (every round's query vs every candidate's full gallery)
# is computed ONCE per animal and cached as a (3, n) matrix. Every aggregation
# POLICY (max / mean / median / quality-filtered-max) is then just a cheap
# reduction over that same cached matrix along the round axis -- so comparing
# five policies costs one similarity pass, not five.

def compute_score_matrices(embeddings: dict[str, np.ndarray]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Returns (tags, {tag: (3, n) matrix}). matrix[i, c] is round i's query
    (animal's photo i) scored against candidate c's gallery -- c's own full 3
    embeddings if c != this animal, or this animal's OTHER 2 embeddings
    (excluding photo i, so it can never match itself) if c == this animal.
    This shape is policy-independent; every aggregation choice reduces it
    along axis 0 (the 3 rounds/query-photos), never axis 1 (candidates)."""
    tags = list(embeddings.keys())
    n = len(tags)
    all_embeds = np.stack([embeddings[t] for t in tags])  # (n, 3, 256)

    matrices: dict[str, np.ndarray] = {}
    for a_idx, tag in enumerate(tags):
        scores_per_round = np.zeros((3, n))
        for i in range(3):
            query = all_embeds[a_idx, i]
            sims = np.einsum("cek,k->ce", all_embeds, query)  # (n, 3)
            sims[a_idx, i] = -np.inf  # exclude self-match against its own query photo
            scores_per_round[i] = sims.max(axis=1)  # max over EACH candidate's own stored embeddings
        matrices[tag] = scores_per_round
    return tags, matrices


def n1_trials_from(tags: list[str], matrices: dict[str, np.ndarray]) -> list[dict]:
    """The fixed N=1 baseline: every individual round, unaggregated, pooled.
    Identical across every policy comparison -- computed once, reused."""
    n = len(tags)
    trials = []
    for a_idx, tag in enumerate(tags):
        m = matrices[tag]  # (3, n)
        for i in range(3):
            per_candidate = m[i]
            ranked = np.argsort(per_candidate)[::-1]
            top1 = ranked[0]
            top1_score = per_candidate[top1]
            second = per_candidate[ranked[1]] if n > 1 else top1_score
            trials.append({
                "animal": tag,
                "round": i,
                "correct": bool(top1 == a_idx),
                "true_score": float(per_candidate[a_idx]),
                "top1_score": float(top1_score),
                "gap": float(top1_score - second),
            })
    return trials


def aggregate_trials(
    tags: list[str],
    matrices: dict[str, np.ndarray],
    reduce_fn,
    valid_rounds: dict[str, list[bool]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Apply one cross-round aggregation policy. reduce_fn takes a (k, n)
    array (k = number of valid rounds for this animal, 1<=k<=3) and returns
    an (n,) vector -- np.max / np.mean / np.median along axis=0 all fit this
    shape directly.

    valid_rounds, if given, restricts which of the 3 rounds participate per
    animal (the quality-filtered policy: a round whose query photo fails the
    real quality gate is dropped before aggregating, not just down-weighted).
    An animal with ZERO valid rounds is skipped and returned separately --
    excluded from the metrics is the honest answer, not a silent 0/default.

    top1/gap are BOTH read off the same aggregated vector for every
    candidate, true animal and every impostor alike -- there is no special
    case for "the true candidate", because at search time nothing here (or
    in production) knows in advance which candidate that is. Confirms the
    gap is computed under one consistent treatment, not mixed ones.
    """
    n = len(tags)
    trials = []
    skipped: list[str] = []
    for a_idx, tag in enumerate(tags):
        m = matrices[tag]  # (3, n)
        rounds = valid_rounds[tag] if valid_rounds is not None else [True, True, True]
        idx = [i for i in range(3) if rounds[i]]
        if not idx:
            skipped.append(tag)
            continue
        agg = reduce_fn(m[idx])  # (n,) -- same reduction, same treatment, every candidate
        ranked = np.argsort(agg)[::-1]
        top1 = ranked[0]
        top1_score = agg[top1]
        second = agg[ranked[1]] if n > 1 else top1_score
        trials.append({
            "animal": tag,
            "correct": bool(top1 == a_idx),
            "true_score": float(agg[a_idx]),
            "top1_score": float(top1_score),
            "gap": float(top1_score - second),
            "n_rounds_used": len(idx),
        })
    return trials, skipped


def _passes_robust(t: dict) -> bool:
    return bool(t["correct"] and t["true_score"] >= MATCH_THRESHOLD and t["gap"] >= GAP_THRESHOLD)


# ── Threshold calibration ─────────────────────────────────────────────────────
#
# decision.go's MATCH_THRESHOLD (0.82) / GAP_THRESHOLD (0.08) were calibrated
# against single-photo (N=1) search. Median aggregation shifts the whole
# score distribution (see the N=1 vs N=3-median numbers already reported:
# true-positive score mean 0.845->0.877, gap mean 0.069->0.078) -- calibrated
# for one distribution, these two constants are not automatically still the
# right cut points for a different one.
#
# This is a real leave-one-out verification calibration, not just re-reading
# the same top-1/gap numbers already reported: for every threshold pair, it
# asks two separate questions across all 189 animals --
#   true accept rate  = how often the REAL animal is top-1 AND clears the bar
#   false accept rate = how often some WRONG animal is top-1 and STILL clears
#                        the bar (a confident wrong MATCH reaching a farmer --
#                        the failure mode that actually matters for safety)
# -- because raising recall by loosening the gate is only a genuine win if it
# doesn't also raise how often a wrong animal clears it.

def aggregated_score_vectors(
    tags: list[str],
    matrices: dict[str, np.ndarray],
    reduce_fn,
) -> dict[str, np.ndarray]:
    """{tag: (n,) aggregated score vector against every candidate}, using the
    same per-animal reduction aggregate_trials uses internally -- exposed
    separately here because a threshold sweep needs the full vector (to find
    top-1 AND the runner-up at each of several hypothetical thresholds), not
    just the single top1/gap pair aggregate_trials reports for the ONE
    threshold already in decision.go."""
    out: dict[str, np.ndarray] = {}
    for tag in tags:
        out[tag] = reduce_fn(matrices[tag])
    return out


def threshold_sweep(
    tags: list[str],
    agg_scores: dict[str, np.ndarray],
    match_thresholds: list[float],
    gap_thresholds: list[float],
) -> list[dict]:
    """For every (match_threshold, gap_threshold) combination: true accept
    rate and false accept rate across all animals in agg_scores. O(T*G*n) --
    trivial at this scale (189 animals, a few hundred grid points)."""
    results = []
    n = len(tags)
    for T in match_thresholds:
        for G in gap_thresholds:
            true_accepts = false_accepts = 0
            for a_idx, tag in enumerate(tags):
                vec = agg_scores.get(tag)
                if vec is None:
                    continue
                ranked = np.argsort(vec)[::-1]
                top1 = ranked[0]
                top1_score = vec[top1]
                second = vec[ranked[1]] if len(vec) > 1 else top1_score
                gap = top1_score - second
                if top1_score >= T and gap >= G:
                    if top1 == a_idx:
                        true_accepts += 1
                    else:
                        false_accepts += 1
            results.append({
                "match_threshold": T,
                "gap_threshold": G,
                "true_accept_rate": true_accepts / n,
                "false_accept_rate": false_accepts / n,
                "true_accepts": true_accepts,
                "false_accepts": false_accepts,
            })
    return results


def report_threshold_sweep(results: list[dict]) -> None:
    current = next(
        (r for r in results if r["match_threshold"] == MATCH_THRESHOLD and r["gap_threshold"] == GAP_THRESHOLD),
        None,
    )
    print("\n" + "=" * 78)
    print("THRESHOLD CALIBRATION (median-aggregated N=3 scores, 189 animals)")
    print("=" * 78)
    if current:
        print(
            f"CURRENT decision.go (match>={MATCH_THRESHOLD}, gap>={GAP_THRESHOLD}): "
            f"true_accept={current['true_accept_rate']:.1%} ({current['true_accepts']}/189)  "
            f"false_accept={current['false_accept_rate']:.1%} ({current['false_accepts']}/189)"
        )

    # Pareto frontier: no other point in the sweep has BOTH a higher true
    # accept rate AND a lower-or-equal false accept rate. Printing only the
    # frontier (instead of the full grid) is what makes "which threshold
    # should we move to" answerable by eye -- every dominated point is by
    # definition a strictly worse choice than something else already on it.
    frontier = []
    for r in results:
        dominated = any(
            o is not r
            and o["true_accept_rate"] >= r["true_accept_rate"]
            and o["false_accept_rate"] <= r["false_accept_rate"]
            and (o["true_accept_rate"], -o["false_accept_rate"]) != (r["true_accept_rate"], -r["false_accept_rate"])
            for o in results
        )
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: (-r["true_accept_rate"], r["false_accept_rate"]))

    print(f"\nPareto frontier ({len(frontier)} of {len(results)} grid points -- "
          f"every other combination is strictly dominated by one of these):")
    print(f"  {'match_thr':>9} {'gap_thr':>8} {'true_accept':>12} {'false_accept':>13}")
    for r in frontier:
        marker = "  <-- current" if current and r is current else ""
        print(
            f"  {r['match_threshold']:>9.3f} {r['gap_threshold']:>8.3f} "
            f"{r['true_accept_rate']:>11.1%} {r['false_accept_rate']:>12.1%}{marker}"
        )

    # Candidate points worth a direct look even when off the frontier -- e.g.
    # a point that respects a Go-side invariant (attributeWeight vs
    # gapThreshold) the pure ROC frontier doesn't know about.
    CANDIDATES = [(0.88, 0.08)]
    print("\nCandidate points (may be off the frontier, printed on request):")
    for mt, gt in CANDIDATES:
        r = next((r for r in results if r["match_threshold"] == mt and r["gap_threshold"] == gt), None)
        if r is None:
            print(f"  ({mt}, {gt}): not in the swept grid")
            continue
        on_frontier = any(r is f for f in frontier)
        print(
            f"  match={mt:.3f} gap={gt:.3f}: true_accept={r['true_accept_rate']:.1%} "
            f"({r['true_accepts']}/189)  false_accept={r['false_accept_rate']:.1%} "
            f"({r['false_accepts']}/189){'  (on frontier)' if on_frontier else '  (dominated)'}"
        )


def _summarize(trials: list[dict], label: str) -> None:
    n = len(trials)
    strict = sum(t["correct"] for t in trials) / n
    robust = sum(_passes_robust(t) for t in trials) / n
    true_scores = np.array([t["true_score"] for t in trials])
    gaps = np.array([t["gap"] for t in trials if t["correct"]])  # gap only meaningful when correct

    print(f"\n{label} (n={n} trials)")
    print(f"  strict match rate (top-1 accuracy):        {strict:.1%}")
    print(f"  robust recall (real MATCH threshold+gap):  {robust:.1%}")
    print(f"  true-positive score:  mean={true_scores.mean():.4f} median={np.median(true_scores):.4f} "
          f"min={true_scores.min():.4f} max={true_scores.max():.4f}")
    if len(gaps):
        print(f"  gap (when correct):   mean={gaps.mean():.4f} median={np.median(gaps):.4f} "
              f"min={gaps.min():.4f} max={gaps.max():.4f}")


def _report_variance(n1_trials: list[dict]) -> None:
    """Per animal: spread between its best and worst single-photo (N=1)
    query score. This is the number that directly measures the field
    failure that motivated this whole investigation -- the same real animal
    scoring very differently a few minutes apart depending on which photo
    was sent."""
    by_animal: dict[str, list[float]] = defaultdict(list)
    for t in n1_trials:
        by_animal[t["animal"]].append(t["true_score"])

    spreads = {tag: (max(v) - min(v)) for tag, v in by_animal.items()}
    vals = np.array(list(spreads.values()))

    print(f"\nPer-animal spread between best and worst single-photo (N=1) true-positive score "
          f"(n={len(vals)} animals):")
    print(f"  mean={vals.mean():.4f} median={np.median(vals):.4f} "
          f"min={vals.min():.4f} max={vals.max():.4f}")
    for pct in (50, 75, 90, 95):
        print(f"  p{pct}={np.percentile(vals, pct):.4f}")
    worst = sorted(spreads.items(), key=lambda kv: -kv[1])[:10]
    print("  10 largest spreads:")
    for tag, spread in worst:
        scores = sorted(by_animal[tag])
        print(f"    {tag}: spread={spread:.4f}  scores={[round(s, 4) for s in scores]}")


def _report_flips(n1_trials: list[dict], agg_trials: list[dict], label: str) -> None:
    """Does this aggregation policy rescue single-photo failures, and does it
    ever make a single-photo SUCCESS worse? Max-aggregation guarantees the
    true-positive SCORE can't drop under that specific policy, but says
    nothing on its own about whether some other animal's query, aggregated,
    could newly outscore it -- so top-1 correctness and the robust-recall
    verdict are checked empirically here, not assumed. Mean/median carry no
    such guarantee at all, which is exactly why this is checked per policy,
    not asserted once."""
    agg_by_animal = {t["animal"]: t for t in agg_trials}
    skipped = 0

    strict_flip_up = strict_flip_down = 0
    robust_flip_up = robust_flip_down = 0
    n_compared = 0
    for t1 in n1_trials:
        ta = agg_by_animal.get(t1["animal"])
        if ta is None:
            skipped += 1
            continue
        n_compared += 1
        if not t1["correct"] and ta["correct"]:
            strict_flip_up += 1
        if t1["correct"] and not ta["correct"]:
            strict_flip_down += 1
        if not _passes_robust(t1) and _passes_robust(ta):
            robust_flip_up += 1
        if _passes_robust(t1) and not _passes_robust(ta):
            robust_flip_down += 1

    print(f"\nFlips, N=1 round vs that animal's {label} aggregate (n={n_compared} trials compared"
          + (f", {skipped} skipped -- animal excluded by this policy)" if skipped else ")"))
    print(f"  strict match:  {strict_flip_up}/{n_compared} single-photo MISSES rescued; "
          f"{strict_flip_down}/{n_compared} single-photo HITS turned into misses")
    print(f"  robust recall: {robust_flip_up}/{n_compared} single-photo FAILS rescued; "
          f"{robust_flip_down}/{n_compared} single-photo PASSES turned into fails")


def _report_dropout_clustering(photos_by_tag: dict, photo_failures: list[tuple[str, str]]) -> None:
    """Do the 43 crop_cattle failures cluster in a few animals (all 3 photos
    of a handful of animals) or spread thinly across many (one bad photo
    each)? These have very different real-world implications: the former
    suggests specific bad registrations, the latter suggests a generic
    per-photo failure rate that will keep costing ~1 in 6 animals."""
    fails_by_tag: dict[str, int] = defaultdict(int)
    for path, _status in photo_failures:
        tag = re.match(r"^(UKDE\w+)_muzzle\d\.jpg$", os.path.basename(path)).group(1)
        fails_by_tag[tag] += 1

    by_count = defaultdict(int)
    for tag in photos_by_tag:
        by_count[fails_by_tag.get(tag, 0)] += 1

    print(f"\nDropout clustering ({len(photo_failures)} failed photos across "
          f"{len(fails_by_tag)} animals, out of {len(photos_by_tag)} total animals):")
    for k in sorted(by_count):
        if k == 0:
            print(f"  {by_count[k]} animals: 0 failed photos (clean)")
        else:
            print(f"  {by_count[k]} animals: {k} failed photo(s) of 3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--yolo-path", default=os.environ.get("YOLO_MODEL_PATH", "yolov8s.pt"))
    parser.add_argument("--dataset-dir", default=os.path.abspath(DATASET_DIR))
    parser.add_argument("--synthetic-selftest", action="store_true",
                         help="Validate this script's OWN logic only. Produces no real evidence.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of animals (debugging only)")
    parser.add_argument("--calibrate", action="store_true",
                         help="Sweep decision.go's MATCH_THRESHOLD/GAP_THRESHOLD against the "
                              "N=3-median score distribution (the policy actually in production now) "
                              "and report the true/false-accept Pareto frontier.")
    args = parser.parse_args()

    photos_by_tag = find_uk_muzzle_photos(args.dataset_dir)
    print(f"Found {len(photos_by_tag)} UK animals with a complete 3-muzzle-photo set "
          f"in {args.dataset_dir}")
    if args.limit:
        photos_by_tag = dict(list(photos_by_tag.items())[: args.limit])
        print(f"(--limit applied: using {len(photos_by_tag)})")

    if args.synthetic_selftest:
        print("\n*** SYNTHETIC SELF-TEST -- validates harness logic only, NOT real accuracy ***")
        embedder = SyntheticEmbedder()
        embeddings = {
            tag: embedder.embed_photos(paths, tag=tag) for tag, paths in photos_by_tag.items()
        }
    else:
        if not args.model_path or not os.path.isfile(args.model_path):
            print(f"\nERROR: no real embedding model found at --model-path={args.model_path!r}.")
            print("This is the exact blocker reported: the trained checkpoint isn't on this machine.")
            print("Re-run with --synthetic-selftest to validate the harness logic only (not a substitute "
                  "for real numbers), or point --model-path at the real weights once available.")
            sys.exit(1)

        embedder = RealEmbedder(args.model_path, args.yolo_path)

        # ── Phase 1: detect every photo individually, so the failure count ──
        # is exact per-photo, not just "this animal had a problem somewhere".
        # Quality-checked here too (same crop, same pass), using the REAL
        # production gate -- not a new threshold invented for this harness.
        total_photos = sum(len(p) for p in photos_by_tag.values())
        crops_by_tag: dict[str, list[np.ndarray]] = {}
        quality_by_tag: dict[str, list[bool]] = {}
        photo_failures: list[tuple[str, str]] = []  # (path, status)
        quality_failures: list[tuple[str, str]] = []  # (path, reason)
        animals_with_failure: set[str] = set()

        print(f"\nRunning crop_cattle on {total_photos} muzzle photos ({len(photos_by_tag)} animals x 3)...")
        for tag, paths in photos_by_tag.items():
            crops = []
            quality = []
            ok = True
            for p in paths:
                crop, status = embedder.detect_crop(p)
                if crop is None:
                    photo_failures.append((p, status))
                    animals_with_failure.add(tag)
                    ok = False
                else:
                    crops.append(crop)
                    q_ok, q_reason = embedder.quality_ok(crop)
                    quality.append(q_ok)
                    if not q_ok:
                        quality_failures.append((p, q_reason))
            if ok:
                crops_by_tag[tag] = crops
                quality_by_tag[tag] = quality

        print(f"\nDetection results: {total_photos - len(photo_failures)}/{total_photos} photos "
              f"cropped successfully, {len(photo_failures)} failed:")
        for p, status in photo_failures:
            print(f"    FAIL {os.path.basename(p)}: {status}")
        if animals_with_failure:
            print(f"  {len(animals_with_failure)} animals have at least one failed photo, "
                  f"excluded from the leave-one-out set entirely (all 3 photos are needed): "
                  f"{sorted(animals_with_failure)}")
        _report_dropout_clustering(photos_by_tag, photo_failures)

        print(f"\nQuality gate (pipeline.quality.quality_check_cv2, BLUR_THRESHOLD=20.0) on the "
              f"{sum(len(c) for c in crops_by_tag.values())} successfully-cropped photos: "
              f"{len(quality_failures)} failed:")
        for p, reason in quality_failures:
            flag = "  <-- UKDEGR071423" if "UKDEGR071423" in p else ""
            print(f"    FAIL {os.path.basename(p)}: {reason}{flag}")
        if not any("UKDEGR071423" in p for p, _ in quality_failures):
            print("    (UKDEGR071423's low-scoring photo passed this quality gate -- "
                  "NOT caught by blur/exposure filtering)")

        # ── Phase 2: batch-embed each animal's 3 crops in one forward pass --
        # same as /register's own embed_batch call, not one-photo-at-a-time.
        embeddings = {}
        for tag, crops in crops_by_tag.items():
            embeddings[tag] = embedder.embed_crops(crops)
        print(f"\n{len(embeddings)}/{len(photos_by_tag)} animals embedded successfully, "
              f"proceeding to evaluation.")

    tags, matrices = compute_score_matrices(embeddings)
    n1_trials = n1_trials_from(tags, matrices)

    print("\n" + "=" * 78)
    print("RESULTS" + ("  [SYNTHETIC SELF-TEST -- not real evidence]" if args.synthetic_selftest else ""))
    print("=" * 78)
    _summarize(n1_trials, "N=1 (current production behavior)")
    if not args.synthetic_selftest:
        _report_variance(n1_trials)

    policies: list[tuple[str, object, dict | None]] = [
        ("N=3 max (already reported)", lambda m: m.max(axis=0), None),
        ("N=3 mean", lambda m: m.mean(axis=0), None),
        ("N=3 median", lambda m: np.median(m, axis=0), None),
    ]
    if not args.synthetic_selftest:
        policies.append(("N=3 quality-filtered max", lambda m: m.max(axis=0), quality_by_tag))

    for label, reduce_fn, valid_rounds in policies:
        agg_trials, skipped = aggregate_trials(tags, matrices, reduce_fn, valid_rounds)
        if skipped:
            print(f"\n[{label}] {len(skipped)} animal(s) had zero valid rounds, excluded: {skipped}")
        _summarize(agg_trials, label)
        if not args.synthetic_selftest:
            _report_flips(n1_trials, agg_trials, label)

    if args.calibrate:
        if args.synthetic_selftest:
            print("\n--calibrate on synthetic data would recalibrate against a fake signal -- refusing.")
        else:
            median_scores = aggregated_score_vectors(tags, matrices, lambda m: np.median(m, axis=0))
            # Grid: 0.02 steps, explicitly including the exact current
            # thresholds (0.82/0.08) so the report can mark "you are here"
            # rather than only showing nearby grid points.
            match_grid = sorted({round(0.60 + 0.02 * i, 2) for i in range(21)} | {MATCH_THRESHOLD})
            gap_grid = sorted({round(0.02 * i, 2) for i in range(13)} | {GAP_THRESHOLD})
            sweep = threshold_sweep(tags, median_scores, match_grid, gap_grid)
            report_threshold_sweep(sweep)


if __name__ == "__main__":
    main()
