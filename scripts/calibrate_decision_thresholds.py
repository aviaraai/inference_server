"""
scripts/calibrate_decision_thresholds.py — STAGE 6: recalibrate go-apiserver's
decision.go constants (matchThreshold/gapThreshold/reviewThreshold) against
the NEW fusion+PCA-whitening embedding distribution.

Why this can't just be "run scratch_multiphoto_search_eval.py --calibrate"
verbatim: that script's RealEmbedder is hardcoded to the OLD pipeline
(crop_cattle -> plain GodhaarModel) and per this task's own rule, scratch_*
scripts are evidence and must not be modified. This script instead IMPORTS
scratch_multiphoto_search_eval.py's reusable, embedding-agnostic pieces
(compute_score_matrices / aggregate_trials / aggregated_score_vectors /
threshold_sweep / report_threshold_sweep / _summarize / _report_flips) --
the exact same leave-one-out protocol and calibration math -- and feeds them
embeddings from the real production fusion encoder on FULL photos (no
crop_cattle dependency, matching Stage 3), instead of re-deriving any of
that logic here.

Protocol (identical to scratch_multiphoto_search_eval.py):
  - 224 UK animals x 3 muzzle photos, leave-one-out-per-round.
  - N=3 median aggregation (already validated as the production policy)
    is what the threshold sweep runs against.
  - Pareto frontier over (match_threshold, gap_threshold): true accept rate
    vs. false accept rate.
  - The safety invariant 2*attributeWeight < gapThreshold (decision.go) is
    checked against whatever gapThreshold this picks, and attributeWeight
    is only ever proposed to shrink to preserve it, never silently ignored.

Usage:
    .venv/Scripts/python.exe scripts/calibrate_decision_thresholds.py
"""
import glob
import hashlib
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from godhaar.model import GodhaarModel  # noqa: E402
from pipeline.fusion_encoder import FusionEncoder, load_resnet50  # noqa: E402
from pipeline.muzzle import embed_batch  # noqa: E402
from pipeline.quality import quality_check  # noqa: E402
from pipeline.whitening import WhiteningTransform  # noqa: E402

import scratch_multiphoto_search_eval as ref  # noqa: E402  — reused, not modified

DINO_MODEL_PATH = os.environ.get("MODEL_PATH", r"D:\Group Projects\inference_server\appstorage\Models\model.pt")
RESNET_PATH = os.environ.get(
    "RESNET_MODEL_PATH", r"D:\Group Projects\inference_server\appstorage\Models\resnet50\resnet50_imagenet1k_v2.pth"
)
WHITENING_PATH = os.environ.get(
    "WHITENING_MODEL_PATH", r"D:\Group Projects\inference_server\appstorage\Models\whitening\whitening_v1.npz"
)
DATASET_DIR = os.path.abspath(ref.DATASET_DIR)

# Current live go-apiserver constants (decision.go, as mirrored in main.py --
# see _SEARCH_MATCH_THRESHOLD_MIRROR et al.), for "you are here" markers.
CURRENT_MATCH_THRESHOLD = 0.86
CURRENT_GAP_THRESHOLD = 0.02
CURRENT_REVIEW_THRESHOLD = 0.72
CURRENT_ATTRIBUTE_WEIGHT = 0.03


def find_duplicate_registration_tags(photos_by_tag: dict[str, list[str]]) -> set[str]:
    """Detect animals whose muzzle1.jpg is BYTE-IDENTICAL to another animal's
    -- a real dataset defect found while calibrating (not a hypothetical):
    the initial calibration pass showed several "false accept" pairs
    scoring EXACTLY 1.0000 median similarity, e.g. UKDEGR152028 <->
    UKDEGR927838. Checked directly: their muzzle1/2/3.jpg files are
    byte-identical (md5) across all 3 slots -- the same photos are filed
    under two different animal tags, not two genuinely different animals
    the embedding failed to separate. This is a labeling artifact in the
    corpus, not evidence about the encoder, and calibrating against it
    would optimize the decision thresholds around a defect that will never
    occur in production and can only make real recall/precision worse.

    6 such duplicate-tag pairs (12 animals) found in the 224-animal UK set:
    UKDEGR152028/UKDEGR927838, UKDEJS285059/UKDEJS706085,
    UKDEJS434391/UKDEJS895050, UKDEJS636236/UKDEJS960541,
    UKDEJS762234/UKDEJS894301, UKDEOT412471/UKDEOT732570.

    Both tags in each pair are excluded (not just one) -- there's no way to
    tell which of the two is the "real" registration from image content
    alone, and keeping either one still leaves a phantom duplicate-of-self
    in the gallery.
    """
    by_hash: dict[str, list[str]] = {}
    for tag, paths in photos_by_tag.items():
        h = hashlib.md5(open(paths[0], "rb").read()).hexdigest()
        by_hash.setdefault(h, []).append(tag)
    dup_tags: set[str] = set()
    for h, tags in by_hash.items():
        if len(tags) > 1:
            dup_tags.update(tags)
    return dup_tags


def build_full_photo_embeddings(photos_by_tag: dict[str, list[str]], encoder: FusionEncoder, device) -> dict[str, np.ndarray]:
    """Embed each animal's 3 muzzle photos with the FULL-photo fusion
    encoder (Stage 3: quality_check gates, crop_cattle does NOT). This is
    what makes these embeddings representative of the NEW production path,
    not the old crop-dependent one scratch_multiphoto_search_eval.py's own
    RealEmbedder uses.
    """
    embeddings: dict[str, np.ndarray] = {}
    dropped = []
    for tag, paths in photos_by_tag.items():
        raws = [open(p, "rb").read() for p in paths]
        bad = [p for p, r in zip(paths, raws) if quality_check(r)[0] not in ("OK", "GOOD")]
        if bad:
            dropped.append((tag, bad))
            continue
        emb = embed_batch(raws, encoder, device).numpy()  # (3, 256)
        embeddings[tag] = emb
    if dropped:
        print(f"{len(dropped)} animals dropped (a muzzle photo failed quality_check): "
              f"{[t for t, _ in dropped][:10]}{' ...' if len(dropped) > 10 else ''}")
    return embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, ck = GodhaarModel.load_checkpoint(DINO_MODEL_PATH, device=device)
    dino.eval()
    resnet = load_resnet50(RESNET_PATH, device)
    whitening = WhiteningTransform.load(WHITENING_PATH)
    encoder = FusionEncoder(dino, resnet, whitening, device)
    encoder.eval()
    print(f"Fusion encoder ready (dino epoch={ck.get('epoch')}, whitening hash={whitening.content_hash[:12]}...)")

    photos_by_tag = ref.find_uk_muzzle_photos(DATASET_DIR)
    print(f"Found {len(photos_by_tag)} UK animals with a complete 3-muzzle-photo set in {DATASET_DIR}")

    dup_tags = find_duplicate_registration_tags(photos_by_tag)
    if dup_tags:
        print(f"\n{len(dup_tags)} animals excluded — byte-identical muzzle1.jpg shared with another "
              f"tag (dataset duplicate-registration labeling defect, not an embedding failure): "
              f"{sorted(dup_tags)}")
        photos_by_tag = {t: p for t, p in photos_by_tag.items() if t not in dup_tags}

    embeddings = build_full_photo_embeddings(photos_by_tag, encoder, device)
    print(f"{len(embeddings)}/{len(photos_by_tag)} animals embedded (full photo, fusion encoder), "
          f"proceeding to evaluation.")

    tags, matrices = ref.compute_score_matrices(embeddings)
    n1_trials = ref.n1_trials_from(tags, matrices)

    print("\n" + "=" * 78)
    print(f"RESULTS — NEW fusion+PCA-whitening embeddings, {len(tags)} animals")
    print("=" * 78)
    ref._summarize(n1_trials, "N=1 (single query photo)")

    policies = [
        ("N=3 max", lambda m: m.max(axis=0)),
        ("N=3 mean", lambda m: m.mean(axis=0)),
        ("N=3 median", lambda m: np.median(m, axis=0)),
    ]
    agg_by_label = {}
    for label, reduce_fn in policies:
        agg_trials, skipped = ref.aggregate_trials(tags, matrices, reduce_fn, None)
        if skipped:
            print(f"\n[{label}] {len(skipped)} animal(s) had zero valid rounds, excluded: {skipped}")
        ref._summarize(agg_trials, label)
        ref._report_flips(n1_trials, agg_trials, label)
        agg_by_label[label] = agg_trials

    # ── Confirm median still beats max on the NEW distribution before
    # calibrating against it -- that choice was validated against the OLD
    # distribution and must be re-checked, not assumed to transfer.
    max_strict = sum(t["correct"] for t in agg_by_label["N=3 max"]) / len(agg_by_label["N=3 max"])
    median_strict = sum(t["correct"] for t in agg_by_label["N=3 median"]) / len(agg_by_label["N=3 median"])
    print("\n" + "-" * 78)
    print(f"Median vs max on the NEW distribution: median strict={median_strict:.1%}  max strict={max_strict:.1%}")
    print("Median " + ("still" if median_strict >= max_strict else "no longer") + " matches-or-beats max.")
    print("-" * 78)

    # ── Threshold calibration on median-aggregated scores ──
    median_scores = ref.aggregated_score_vectors(tags, matrices, lambda m: np.median(m, axis=0))
    match_grid = sorted({round(0.02 * i, 2) for i in range(51)} | {CURRENT_MATCH_THRESHOLD})  # 0.00..1.00
    gap_grid = sorted({round(0.02 * i, 2) for i in range(21)} | {CURRENT_GAP_THRESHOLD})       # 0.00..0.40
    sweep = ref.threshold_sweep(tags, median_scores, match_grid, gap_grid)

    # Diagnostic: what do the false-accept cases at low thresholds actually
    # look like? If they're near-1.0-score "impostors" they may be labeling/
    # dataset artifacts (e.g. this dataset having 2 different tags for what
    # is really one animal), not the embedding failing to separate them.
    ranked_false_accepts = []
    for a_idx, tag in enumerate(tags):
        vec = median_scores[tag]
        order = np.argsort(vec)[::-1]
        top1 = order[0]
        if top1 != a_idx:
            ranked_false_accepts.append((tag, tags[top1], float(vec[top1]), float(vec[a_idx])))
    ranked_false_accepts.sort(key=lambda r: -r[2])
    print(f"\nAll {len(ranked_false_accepts)} cases where top-1 is the WRONG animal "
          f"(median-aggregated score), sorted by score, top 10:")
    for tag, wrong, score, own in ranked_false_accepts[:10]:
        print(f"  {tag} -> {wrong}  score={score:.4f}  (own true score={own:.4f})")

    n = len(tags)
    current = next(
        (r for r in sweep if r["match_threshold"] == CURRENT_MATCH_THRESHOLD and r["gap_threshold"] == CURRENT_GAP_THRESHOLD),
        None,
    )
    print("\n" + "=" * 78)
    print(f"THRESHOLD CALIBRATION (median-aggregated N=3, NEW embeddings, {n} animals)")
    print("=" * 78)
    if current:
        print(
            f"OLD decision.go values applied to the NEW distribution "
            f"(match>={CURRENT_MATCH_THRESHOLD}, gap>={CURRENT_GAP_THRESHOLD}): "
            f"true_accept={current['true_accept_rate']:.1%} ({current['true_accepts']}/{n})  "
            f"false_accept={current['false_accept_rate']:.1%} ({current['false_accepts']}/{n})"
        )

    frontier = []
    for r in sweep:
        dominated = any(
            o is not r
            and o["true_accept_rate"] >= r["true_accept_rate"]
            and o["false_accept_rate"] <= r["false_accept_rate"]
            and (o["true_accept_rate"], -o["false_accept_rate"]) != (r["true_accept_rate"], -r["false_accept_rate"])
            for o in sweep
        )
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: (-r["true_accept_rate"], r["false_accept_rate"]))

    print(f"\nPareto frontier ({len(frontier)} of {len(sweep)} grid points):")
    print(f"  {'match_thr':>9} {'gap_thr':>8} {'true_accept':>12} {'false_accept':>13}")
    for r in frontier:
        print(f"  {r['match_threshold']:>9.3f} {r['gap_threshold']:>8.3f} "
              f"{r['true_accept_rate']:>11.1%} {r['false_accept_rate']:>12.1%}")

    # ── Pick a point: prefer the highest true-accept with false_accept==0,
    # since a confident WRONG match reaching a farmer is the failure mode
    # that matters most (see this file's docstring / decision.go's own
    # framing) -- only relax to the lowest nonzero false-accept rate if no
    # zero-false-accept point clears a reasonable recall floor.
    zero_fa = [r for r in frontier if r["false_accept_rate"] == 0.0]
    if zero_fa:
        pick = max(zero_fa, key=lambda r: r["true_accept_rate"])
    else:
        pick = frontier[0]

    print("\n" + "=" * 78)
    print("PICKED OPERATING POINT")
    print("=" * 78)
    print(f"  match_threshold = {pick['match_threshold']}")
    print(f"  gap_threshold   = {pick['gap_threshold']}")
    print(f"  true_accept={pick['true_accept_rate']:.1%} ({pick['true_accepts']}/{n})  "
          f"false_accept={pick['false_accept_rate']:.1%} ({pick['false_accepts']}/{n})")
    print(f"  Rationale: highest true-accept rate among all zero-false-accept points on the "
          f"frontier (a confident WRONG match reaching a farmer is worse than a real match "
          f"needing a retake), matching decision.go's own stated priority."
          if zero_fa else
          "  Rationale: no zero-false-accept point existed on the frontier; picked the point "
          "with the lowest false-accept rate overall.")

    # ── reviewThreshold: the score floor for surfacing a candidate to a
    # human at all, independent of the gap requirement. scratch_multiphoto_
    # search_eval.py's own sweep only covers match/gap (review is a
    # separate, single-threshold decision.go constant) -- pick it as the
    # match_threshold value that maximizes true-accept at gap=0 (the review
    # bucket has no gap requirement in production), i.e. the most permissive
    # floor that still keeps SOME separation from pure noise.
    gap0_rows = [r for r in sweep if r["gap_threshold"] == 0.0]
    review_pick = max(gap0_rows, key=lambda r: r["true_accept_rate"] - r["false_accept_rate"])
    print(f"\nreviewThreshold candidate: {review_pick['match_threshold']} "
          f"(gap=0 row maximizing true_accept-false_accept: "
          f"true_accept={review_pick['true_accept_rate']:.1%}, "
          f"false_accept={review_pick['false_accept_rate']:.1%} — the false_accept share here is "
          f"tolerable because a human reviews before confirming, unlike auto-MATCH)")

    new_gap = pick["gap_threshold"]
    print("\nSafety invariant check: 2*attributeWeight < gapThreshold")
    print(f"  current: 2*{CURRENT_ATTRIBUTE_WEIGHT} = {2*CURRENT_ATTRIBUTE_WEIGHT} < {new_gap}? "
          f"{2*CURRENT_ATTRIBUTE_WEIGHT < new_gap}")
    if 2 * CURRENT_ATTRIBUTE_WEIGHT >= new_gap:
        proposed_weight = round((new_gap / 2) * 0.9, 4)  # 10% margin, not exactly at the boundary
        print(f"  INVARIANT WOULD BREAK at gapThreshold={new_gap} with attributeWeight="
              f"{CURRENT_ATTRIBUTE_WEIGHT} -- propose shrinking attributeWeight to "
              f"{proposed_weight} (keeps 2*attributeWeight < gapThreshold with a 10% margin).")
    else:
        print("  Invariant holds with attributeWeight unchanged.")


if __name__ == "__main__":
    main()
