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
data. D:\\Group Projects\\godhaar-all-images\\godhaar (222 real Uttarakhand
animals, on disk locally) is what's available for this pass.

DRY-RUN BY DEFAULT (no --out write) unless --execute is passed, mirroring
scripts/backfill_muzzle_cache.py's convention for anything that writes a
new artifact under /appstorage.
"""
import argparse
import collections
import glob
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


def l2norm(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


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
    print(f"{len(paths)} candidate images found under {args.corpus_dir}", flush=True)

    dino, ck = GodhaarModel.load_checkpoint(args.dino_model_path, device=device)
    dino.eval()
    resnet = load_resnet50(args.resnet_model_path, device)
    encoder = FusionEncoder(dino, resnet, whitening=None, device=device)
    print(f"encoders loaded (dino epoch={ck.get('epoch')})", flush=True)

    fused_list, tags, kept_paths = [], [], []
    t0 = time.time()
    for n, p in enumerate(paths):
        raw = open(p, "rb").read()
        if quality_check(raw)[0] not in ("OK", "GOOD"):
            continue
        fused = encoder.embed_fused_raw([raw])[0]
        fused_list.append(fused)
        tags.append(os.path.basename(p).split("_muzzle")[0])
        kept_paths.append(p)
        if (n + 1) % 150 == 0:
            print(f"  embedded {n+1}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

    F = np.array(fused_list, dtype=np.float32)
    tags = np.array(tags)
    print(f"embedded {len(F)} images across {len(set(tags.tolist()))} identities ({time.time()-t0:.0f}s)", flush=True)

    ids = sorted(set(tags.tolist()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_fit_ids = max(1, int(len(ids) * args.fit_fraction))
    fit_ids = set(ids[:n_fit_ids])
    eval_ids = set(ids[n_fit_ids:])
    cnt = collections.Counter(tags.tolist())
    usable = {t for t, c in cnt.items() if c >= 2}

    fit_mask = np.array([t in fit_ids for t in tags])
    fit_paths = [p for p, keep in zip(kept_paths, fit_mask) if keep]
    print(f"fitting PCA on {fit_mask.sum()} images from {len(fit_ids)} identities "
          f"(held out: {len(eval_ids)} identities for validation)", flush=True)

    fit_result = WhiteningTransform.fit(F[fit_mask], n_components=args.n_components)
    print(f"fit rank={fit_result['fit_rank']}, requested n_components={args.n_components}", flush=True)

    # ── Validate on held-out identities, exactly as scratch_ablation_pca.py does:
    #    project the FULL corpus (fit + held-out) through this whitening, then
    #    only SCORE queries whose identity is in eval_ids. This never lets a
    #    held-out identity's data influence mean/components -- fit_mask alone
    #    controlled that, above.
    mean = fit_result["mean"]; components = fit_result["components"]
    centered = F - mean[None, :]
    projected = centered @ components.T
    whitened = l2norm(projected)
    metrics = loo(whitened, tags, eval_ids, usable)
    print("=" * 66)
    print(f"HELD-OUT VALIDATION ({metrics['n']} queries, identities absent from the fit):")
    print(f"  top-1  {metrics['top1']}")
    print(f"  top-5  {metrics['top5']}")
    print(f"  top-10 {metrics['top10']}")
    print(f"  sep    {metrics['sep']}")
    print("=" * 66)

    if not args.execute:
        print("\nDRY RUN — artifact NOT written. Pass --execute to write it to:")
        print(f"  {args.out}")
        return

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
        fit_image_list=np.array(fit_paths),
        fit_timestamp=np.array(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        held_out_eval=np.array(json.dumps(metrics)),
    )
    print(f"\nWritten: {out_path}")
    print(f"content_hash={fit_result['content_hash']}")


if __name__ == "__main__":
    main()
