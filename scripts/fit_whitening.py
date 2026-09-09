"""
scripts/fit_whitening.py — STAGE 4: fit the frozen PCA-whitening matrix for
the fusion encoder (pipeline/fusion_encoder.py, pipeline/whitening.py).

⚠️ THE MATRIX IS FIT ONCE, THEN FROZEN. Refitting silently changes the
embedding space and invalidates every vector already stored in the FAISS
index. Running this script again produces a NEW artifact with a NEW content
hash -- pipeline/whitening.py's loader refuses to serve it against an index
built with a different hash (see WHITENING_MODEL_PATH / expected_hash), and
scripts/reindex_gallery.py must re-run against the new artifact before any
traffic is served with it. There is no "quietly refresh the PCA" workflow;
if you're re-fitting, you are planning a migration, not a tune-up.

Hard requirements enforced here, not just documented:
  - n_components must not exceed the fit set's own numerical rank. In the
    reference sweep (scratch_ablation_pca.py), 512/1024 components against
    a ~312-image fit set collapsed to 16.9% top-1 -- components past the
    rank divide by a ~zero eigenvalue and are noise, not signal.
    pipeline/whitening.py's WhiteningTransform.fit() raises if you ask for
    more than the rank supports; this script does not try to catch and
    silently reduce that -- it fails loudly instead.
  - Validated on identities held OUT of the fit set (same protocol
    scratch_ablation_pca.py uses), and those held-out numbers are written
    INTO the artifact (`held_out_eval`) so the artifact is self-documenting
    about how it was validated, not just what it contains.
  - The exact image list the PCA was fit on is recorded (`fit_image_list`),
    so a later audit can tell whether a specific animal's registration
    photo influenced the whitening space it's stored in.

⚠️ DATA-LEAK FIX (this revision): the corpus this script globs from
(D:\\Group Projects\\godhaar-all-images) has TWO overlapping folders --
godhaar/ and Godhaar_aron/ -- covering 193 of 248 identities in both, with
the SAME photo file present byte-for-byte under both folders for most of
those animals (verified: 74.2% of the combined 1,207-image corpus are exact
SHA-256 duplicates of another image under the same identity tag). The
identity-level fit/eval split (unchanged, still correct in principle) does
NOT stop a duplicate from sitting on BOTH sides of a held-out query's own
comparison: 83.6% of held-out eval images had an exact byte-twin still in
their own gallery, which scores cosine 1.000 against itself by construction
-- not a property of the encoder. This inflated the first-shipped artifact's
self-reported held_out_eval (top-1 95.7%) well above the corrected number
(top-1 87.7%, see scratch_dedup_recheck.py, kept unmodified as the evidence
for this fix). The fix: hash and collapse byte-identical images to one
representative BEFORE the fit/eval split (and before the PCA fit itself --
duplicate rows also double-weight those photos in the fit covariance, a
smaller effect than the eval-side leak but still wrong to leave in). See
`deduplicate_by_content()` below.

Usage
-----
    fit_whitening.py --corpus-dir <dir> --glob "UKDE*_muzzle*.jpg" \\
        --n-components 256 --out /appstorage/Models/whitening/whitening_v1.npz

`--corpus-dir` should be the LARGEST available muzzle-image corpus.
Production should fit on the real registered gallery (export it via
go-apiserver's debug endpoint / object storage, same as
scripts/backfill_muzzle_cache.py's approach, then point --corpus-dir at the
downloaded images) -- this script has no DB/GCS access itself, same
limitation as every other inference_server script that touches production
data. D:\\Group Projects\\godhaar-all-images (both godhaar/ and Godhaar_aron/,
treated as ONE overlapping corpus -- see the data-leak note above) is what's
available for this pass.

DRY-RUN BY DEFAULT (no --out write) unless --execute is passed, mirroring
scripts/backfill_muzzle_cache.py's convention for anything that writes a
new artifact under /appstorage.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from godhaar.model import GodhaarModel
from pipeline.fusion_encoder import FusionEncoder, load_resnet50
from pipeline.quality import quality_check
from pipeline.whitening import WhiteningTransform

DEFAULT_DINO_MODEL_PATH = r"D:\Group Projects\inference_server\appstorage\Models\model.pt"
DEFAULT_RESNET_PATH = r"D:\Group Projects\inference_server\appstorage\Models\resnet50\resnet50_imagenet1k_v2.pth"


def tag_of(path: str) -> str:
    return os.path.basename(path).split("_muzzle")[0]


def l2norm(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def deduplicate_by_content(paths: list[str]) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """STAGE A: hash every image (SHA-256 of raw bytes) and collapse each
    group of byte-identical images to ONE representative path (the lowest
    sorted path in the group -- deterministic, arbitrary which). This must
    run BEFORE any fit/eval split and BEFORE the PCA fit itself: see this
    module's docstring for why leaving duplicates in either leaks eval
    signal (a query retrieving its own byte-twin) and biases the fit
    covariance (a duplicated photo is double-weighted).

    Returns (representative_paths sorted, hash -> [paths in that group],
    path -> hash) so the caller can log full auditability, not just apply
    a silent transformation.
    """
    hash_to_paths: dict[str, list[str]] = collections.defaultdict(list)
    path_to_hash: dict[str, str] = {}
    for p in paths:
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        hash_to_paths[h].append(p)
        path_to_hash[p] = h
    representatives = sorted(min(v) for v in hash_to_paths.values())
    return representatives, dict(hash_to_paths), path_to_hash


def loo(E, tags, only_ids, usable_ids):
    """Same leave-one-out protocol as scratch_ablation_pca.py."""
    S = E @ E.T
    ranks, g, m = [], [], []
    for i in range(len(E)):
        if tags[i] not in usable_ids or tags[i] not in only_ids:
            continue
        per = {}
        for j in range(len(E)):
            if j == i:
                continue
            per[tags[j]] = max(per.get(tags[j], -2.0), float(S[i, j]))
        order = sorted(per.items(), key=lambda kv: -kv[1])
        ranks.append([a for a, _ in order].index(tags[i]) + 1)
        g.append(per[tags[i]])
        m.append(max(v for a, v in per.items() if a != tags[i]))
    if not ranks:
        return {"n": 0, "top1": None, "top5": None, "top10": None, "sep": None}
    r = np.array(ranks); g = np.array(g); m = np.array(m)
    return {
        "n": len(r),
        "top1": float(np.mean(r <= 1) * 100),
        "top5": float(np.mean(r <= 5) * 100),
        "top10": float(np.mean(r <= 10) * 100),
        "sep": float(g.mean() - m.mean()),
    }


def fit_and_validate(F: np.ndarray, tags: np.ndarray, fit_ids: set, eval_ids: set,
                      n_components: int, label: str) -> tuple[dict, dict]:
    """Fit PCA on `fit_ids`' rows of F, validate leave-one-out on `eval_ids`'
    rows against the FULL F (fit+eval) as gallery. Returns (fit_result,
    metrics). Shared by both the OLD (non-deduplicated) and NEW
    (deduplicated) runs below so the two numbers come from identical code,
    differing only in which corpus (F, tags) was passed in.
    """
    cnt = collections.Counter(tags.tolist())
    usable = {t for t, c in cnt.items() if c >= 2}
    fit_mask = np.array([t in fit_ids for t in tags])

    fit_result = WhiteningTransform.fit(F[fit_mask], n_components=n_components)
    print(f"[{label}] fitting PCA on {int(fit_mask.sum())} images from {len(fit_ids)} identities "
          f"(held out: {len(eval_ids)} identities), fit rank={fit_result['fit_rank']}, "
          f"requested n_components={n_components}", flush=True)

    mean = fit_result["mean"]; components = fit_result["components"]
    centered = F - mean[None, :]
    projected = centered @ components.T
    whitened = l2norm(projected)
    metrics = loo(whitened, tags, eval_ids, usable)
    print(f"[{label}] HELD-OUT VALIDATION ({metrics['n']} queries, identities absent from the fit): "
          f"top-1={metrics['top1']}  top-5={metrics['top5']}  top-10={metrics['top10']}  sep={metrics['sep']}",
          flush=True)
    return fit_result, metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", required=True, help="Directory of muzzle images to fit on")
    p.add_argument("--glob", default="*_muzzle*.jpg", help="Filename glob within --corpus-dir")
    p.add_argument("--tag-fn", default="parent_dir_or_prefix",
                    help="How to derive an animal identity tag from a filename: "
                         "'parent_dir_or_prefix' splits on '_muzzle' (this repo's convention)")
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--fit-fraction", type=float, default=0.7,
                    help="Fraction of identities used to FIT the PCA; the rest are held out "
                         "for validation only (never used to compute mean/components)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dino-model-path", default=DEFAULT_DINO_MODEL_PATH)
    p.add_argument("--resnet-model-path", default=DEFAULT_RESNET_PATH)
    p.add_argument("--out", default="/appstorage/Models/whitening/whitening_v1.npz")
    p.add_argument("--execute", action="store_true", help="Actually write the artifact (default: dry-run)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.corpus_dir, "**", args.glob), recursive=True))
    paths += sorted(glob.glob(os.path.join(args.corpus_dir, args.glob)))
    paths = sorted(set(paths))
    if not paths:
        print(f"No images matched {args.glob!r} under {args.corpus_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(paths)} raw candidate paths found under {args.corpus_dir}", flush=True)

    # ── STAGE A: de-duplicate by content, BEFORE any embedding/fit/eval split ──
    print("Hashing corpus for exact-duplicate detection (SHA-256)...", flush=True)
    representatives, hash_to_paths, path_to_hash = deduplicate_by_content(paths)
    dup_groups = {h: v for h, v in hash_to_paths.items() if len(v) > 1}
    n_redundant_copies = sum(len(v) - 1 for v in dup_groups.values())
    identities_raw = {tag_of(p) for p in paths}
    identities_dedup = {tag_of(p) for p in representatives}
    print(
        f"De-duplication: {len(paths)} raw paths -> {len(representatives)} distinct-content images "
        f"({len(dup_groups)} duplicate groups, {n_redundant_copies} redundant byte-identical copies "
        f"collapsed), spanning {len(identities_dedup)} identities (of {len(identities_raw)} in the raw corpus).",
        flush=True,
    )
    cross_identity_dupes = [
        (h, v) for h, v in dup_groups.items() if len({tag_of(p) for p in v}) > 1
    ]
    if cross_identity_dupes:
        print(
            f"WARNING: {len(cross_identity_dupes)} duplicate-content group(s) have the SAME photo filed "
            f"under DIFFERENT identity tags (separate from this fix -- not auto-resolved here, just "
            f"surfaced for visibility, same class of issue Stage 6 found and excluded independently "
            f"in its own dataset):",
            flush=True,
        )
        for h, v in cross_identity_dupes[:10]:
            print(f"    {sorted({tag_of(p) for p in v})}: {[os.path.basename(p) for p in v]}", flush=True)

    dino, ck = GodhaarModel.load_checkpoint(args.dino_model_path, device=device)
    dino.eval()
    resnet = load_resnet50(args.resnet_model_path, device)
    encoder = FusionEncoder(dino, resnet, whitening=None, device=device)
    print(f"encoders loaded (dino epoch={ck.get('epoch')})", flush=True)

    # Embed only the DISTINCT-CONTENT representatives -- a duplicate's
    # embedding is identical to its representative's by construction
    # (FusionEncoder is deterministic on identical bytes, see
    # pipeline/tests/test_fusion_encoder_determinism.py), so there is no
    # need to pay for a second forward pass on the same pixels.
    fused_by_hash: dict[str, np.ndarray] = {}
    quality_ok_by_hash: dict[str, bool] = {}
    t0 = time.time()
    for n, p in enumerate(representatives):
        h = path_to_hash[p]
        raw = open(p, "rb").read()
        ok = quality_check(raw)[0] in ("OK", "GOOD")
        quality_ok_by_hash[h] = ok
        if ok:
            fused_by_hash[h] = encoder.embed_fused_raw([raw])[0]
        if (n + 1) % 150 == 0:
            print(f"  embedded {n+1}/{len(representatives)} distinct images ({time.time()-t0:.0f}s)", flush=True)
    print(f"embedded {len(fused_by_hash)}/{len(representatives)} distinct-content images "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── Build the DEDUPLICATED corpus (one row per distinct content) ──
    kept_paths_dedup = [p for p in representatives if quality_ok_by_hash[path_to_hash[p]]]
    F_dedup = np.array([fused_by_hash[path_to_hash[p]] for p in kept_paths_dedup], dtype=np.float32)
    tags_dedup = np.array([tag_of(p) for p in kept_paths_dedup])

    # ── Build the OLD, NON-DEDUPLICATED corpus for side-by-side comparison
    #    (Stage B) -- every original raw path, quality-gated and embedded via
    #    its content hash's cached result. This reproduces exactly what the
    #    pre-fix script measured (duplicates padding both the fit set and the
    #    eval gallery), without paying to re-run the encoder on identical bytes.
    kept_paths_full = [p for p in paths if quality_ok_by_hash[path_to_hash[p]]]
    F_full = np.array([fused_by_hash[path_to_hash[p]] for p in kept_paths_full], dtype=np.float32)
    tags_full = np.array([tag_of(p) for p in kept_paths_full])

    print(f"\nDEDUPLICATED corpus: {len(F_dedup)} images, {len(set(tags_dedup.tolist()))} identities")
    print(f"NON-DEDUPLICATED (old, buggy) corpus: {len(F_full)} images, {len(set(tags_full.tolist()))} identities\n")

    # Identity split computed ONCE, from the full identity set (dedup never
    # removes an identity, only redundant images of one) -- applying the
    # SAME fit_ids/eval_ids to both runs below isolates the dedup effect
    # from any difference in which identities were held out.
    ids = sorted(set(tags_full.tolist()) | set(tags_dedup.tolist()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_fit_ids = max(1, int(len(ids) * args.fit_fraction))
    fit_ids = set(ids[:n_fit_ids])
    eval_ids = set(ids[n_fit_ids:])

    print("=" * 78)
    print("OLD vs NEW — held-out validation, same identity split, same encoder")
    print("=" * 78)
    _old_fit_result, old_metrics = fit_and_validate(
        F_full, tags_full, fit_ids, eval_ids, args.n_components, label="OLD (non-deduplicated, buggy)"
    )
    fit_result, metrics = fit_and_validate(
        F_dedup, tags_dedup, fit_ids, eval_ids, args.n_components, label="NEW (deduplicated, corrected)"
    )
    print("=" * 78)
    print(f"SUMMARY: top-1 {old_metrics['top1']:.1f}% (old, inflated by duplicate self-matches) "
          f"-> {metrics['top1']:.1f}% (new, corrected)   "
          f"sep {old_metrics['sep']:+.4f} -> {metrics['sep']:+.4f}")
    print("=" * 78)

    baseline_top1 = 58.2  # production baseline (plain DINOv2 + YOLO crop), see CLAUDE.md / task description
    if metrics["top1"] is None or metrics["top1"] < baseline_top1 or metrics["sep"] is None or metrics["sep"] < 0:
        print(
            f"\nSTOPPING: corrected held-out top-1 ({metrics['top1']}) is below the production baseline "
            f"({baseline_top1}%) or separation ({metrics['sep']}) is negative. Something is wrong with "
            f"this fit -- refusing to write an artifact.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.execute:
        print("\nDRY RUN — artifact NOT written. Pass --execute to write it to:")
        print(f"  {args.out}")
        return

    fit_paths_dedup = [p for p, t in zip(kept_paths_dedup, tags_dedup) if t in fit_ids]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(
            f"REFUSING to overwrite an existing whitening artifact at {out_path}. "
            f"Refitting invalidates every embedding stored under it -- move the "
            f"old artifact aside (versioned filename) and update "
            f"WHITENING_MODEL_PATH deliberately, together with re-running "
            f"scripts/reindex_gallery.py, if this refit is intentional.",
            file=sys.stderr,
        )
        sys.exit(1)

    np.savez_compressed(
        out_path,
        mean=fit_result["mean"],
        components=fit_result["components"],
        eigenvalues=fit_result["eigenvalues"],
        fit_rank=np.int64(fit_result["fit_rank"]),
        content_hash=fit_result["content_hash"],
        fit_image_list=np.array(fit_paths_dedup),
        fit_timestamp=np.array(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        held_out_eval=np.array(json.dumps(metrics)),
    )
    print(f"\nWritten: {out_path}")
    print(f"content_hash={fit_result['content_hash']}")


if __name__ == "__main__":
    main()
