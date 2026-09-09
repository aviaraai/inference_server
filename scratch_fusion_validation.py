"""scratch_fusion_validation.py — STAGE 1 GATE for the fusion+PCA-whitening
embedding upgrade.

Reproduces scratch_ablation_fusion.py / scratch_ablation_pca.py's numbers
using the REAL production modules (pipeline/fusion_encoder.py,
pipeline/whitening.py, pipeline/muzzle.py::embed_batch), not the scratch
scripts' inlined copies. If this does not land within ~3 points of the
scratch scripts' numbers, per the task's explicit instruction: STOP, do not
proceed to Stage 2+.

Reference numbers (scratch_ablation_pca.py, 222-animal gallery, identity-
disjoint PCA fit, 256-d whitened):
    top-1 ~81-83%   top-5 ~94%   top-10 ~96%   sep(genuine-impostor) ~+0.18-0.19

Protocol (matches scratch_ablation_pca.py exactly):
  1. Embed every UKDE*_muzzle*.jpg that passes quality_check with the REAL
     FusionEncoder.embed_fused_raw() (DINOv2@518 + ResNet50@384 + @448,
     concat, L2) -- no whitening yet, no YOLO crop (production no longer
     crops before embedding -- see Stage 3).
  2. Split identities 50/50 (seed=0, same as scratch_ablation_pca.py).
  3. Fit whitening (pipeline/whitening.py's WhiteningTransform.fit) on the
     FIT half only, save it as a real .npz artifact, load it back through
     WhiteningTransform.load() (exercises the hash-check loader for real).
  4. Build a real FusionEncoder with that loaded whitening and re-embed
     every image through pipeline.muzzle.embed_batch() (the actual
     production entrypoint) -- this is the code path being validated.
  5. Leave-one-out retrieval against the full 222-animal gallery, scored
     only on EVAL-half queries (identities absent from the PCA fit).
"""
import collections
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from godhaar.model import GodhaarModel
from pipeline.fusion_encoder import FusionEncoder, load_resnet50
from pipeline.muzzle import embed_batch
from pipeline.quality import quality_check
from pipeline.whitening import WhiteningTransform
from pipeline.yolo_crop import crop_cattle, load_yolo

DATA = r"D:\Group Projects\godhaar-all-images\godhaar"
DINO_MODEL_PATH = r"D:\Group Projects\inference_server\appstorage\Models\model.pt"
RESNET_PATH = r"D:\Group Projects\inference_server\appstorage\Models\resnet50\resnet50_imagenet1k_v2.pth"
YOLO_PATH = r"D:\Group Projects\inference_server\appstorage\Models\yolov8s.pt"
N_COMPONENTS = 256
SCRATCH_ARTIFACT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scratch_fusion_validation_whitening.npz"
)

REFERENCE = {"top1": (81.0, 83.0), "top5": (93.0, 95.0), "top10": (95.0, 97.0), "sep": (0.15, 0.22)}
TOLERANCE_PTS = 3.0


def l2norm(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def loo(E, tags, only_ids, usable_ids):
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
    r = np.array(ranks); g = np.array(g); m = np.array(m)
    return len(r), np.mean(r <= 1) * 100, np.mean(r <= 5) * 100, np.mean(r <= 10) * 100, float(g.mean() - m.mean())


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)

    dino, ck = GodhaarModel.load_checkpoint(DINO_MODEL_PATH, device=dev)
    dino.eval()
    resnet = load_resnet50(RESNET_PATH, dev)
    load_yolo(YOLO_PATH)
    print(f"dino epoch={ck.get('epoch')}, resnet50 loaded from local weights", flush=True)

    fusion_no_whitening = FusionEncoder(dino, resnet, whitening=None, device=dev)

    all_paths = sorted(__import__("glob").glob(os.path.join(DATA, "UKDE*_muzzle*.jpg")))
    print(f"{len(all_paths)} muzzle photos found", flush=True)

    # Population gate matches scratch_ablation_fusion.py EXACTLY: quality_check
    # must pass AND crop_cattle must succeed (crop_cattle's result is used only
    # to select the population here, matching the reference scripts -- the
    # embedding itself still goes in on the FULL uncropped photo either way,
    # per the target pipeline). This is what makes the Stage 1 numbers a true
    # apples-to-apples reproduction of the scratch-script reference; production
    # itself (Stage 3) no longer requires crop_cattle to succeed at all, which
    # is a separate, expected, and looser population than this gate check uses.
    paths = []
    for p in all_paths:
        raw = open(p, "rb").read()
        if quality_check(raw)[0] not in ("OK", "GOOD"):
            continue
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        crop, _det, _conf = crop_cattle(bgr)
        if crop is None:
            continue
        paths.append(p)
    print(f"{len(paths)} images pass quality_check AND crop_cattle (reference population)", flush=True)

    fused_list, tags = [], []
    t0 = time.time()
    for n, p in enumerate(paths):
        raw = open(p, "rb").read()
        fused = fusion_no_whitening.embed_fused_raw([raw])  # (1, 4352) — NO crop, whole photo
        fused_list.append(fused[0])
        tags.append(os.path.basename(p).split("_muzzle")[0])
        if (n + 1) % 150 == 0:
            print(f"  fused {n+1}/{len(paths)} {time.time()-t0:.0f}s", flush=True)

    F = np.array(fused_list, dtype=np.float32)
    tags = np.array(tags)
    print(f"embedded {len(F)} images, {len(set(tags.tolist()))} animals ({time.time()-t0:.0f}s)", flush=True)

    ids = sorted(set(tags.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    fit_ids = set(ids[: len(ids) // 2])
    eval_ids = set(ids[len(ids) // 2:])
    cnt = collections.Counter(tags.tolist())
    usable = {t for t, c in cnt.items() if c >= 2}

    fit_mask = np.array([t in fit_ids for t in tags])
    fit_result = WhiteningTransform.fit(F[fit_mask], n_components=N_COMPONENTS)
    print(f"whitening fit on {fit_mask.sum()} images, rank={fit_result['fit_rank']}", flush=True)

    np.savez_compressed(
        SCRATCH_ARTIFACT,
        mean=fit_result["mean"], components=fit_result["components"],
        eigenvalues=fit_result["eigenvalues"], fit_rank=np.int64(fit_result["fit_rank"]),
        content_hash=fit_result["content_hash"],
        fit_image_list=np.array(sorted(fit_ids)),
        fit_timestamp=np.array(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        held_out_eval=np.array(json.dumps({})),
    )
    whitening = WhiteningTransform.load(SCRATCH_ARTIFACT)  # real loader, real hash check

    fusion = FusionEncoder(dino, resnet, whitening=whitening, device=dev)

    # ── Re-embed through the REAL production entrypoint: embed_batch() ──
    embs = []
    for n, p in enumerate(paths):
        raw = open(p, "rb").read()
        e = embed_batch([raw], fusion, dev).numpy()[0]  # (256,), production call
        embs.append(e)
    E = np.array(embs, dtype=np.float32)
    assert E.shape == F.shape[:1] + (256,), f"unexpected shape {E.shape}"
    E = l2norm(E)

    n, top1, top5, top10, sep = loo(E, tags, eval_ids, usable)
    print("=" * 70)
    print(f"STAGE 1 — real production code path, {n} held-out queries (identities absent from PCA fit)")
    print(f"  top-1  {top1:5.1f}%   (reference {REFERENCE['top1']})")
    print(f"  top-5  {top5:5.1f}%   (reference {REFERENCE['top5']})")
    print(f"  top-10 {top10:5.1f}%  (reference {REFERENCE['top10']})")
    print(f"  sep    {sep:+.4f}     (reference {REFERENCE['sep']})")
    print("=" * 70)

    def within(v, lo_hi):
        lo, hi = lo_hi
        return (lo - TOLERANCE_PTS) <= v <= (hi + TOLERANCE_PTS)

    ok = within(top1, REFERENCE["top1"]) and within(top5, REFERENCE["top5"]) and within(top10, REFERENCE["top10"])
    if not ok:
        print("STAGE 1 GATE: FAILED — reproduction is more than ~3 points off the scratch-script reference.")
        print("STOPPING per task instructions. Do not proceed to Stage 2+ until this is resolved.")
        sys.exit(1)
    print("STAGE 1 GATE: PASSED — real code path reproduces the scratch-script numbers within tolerance.")

    json.dump(
        {"n_queries": n, "top1": top1, "top5": top5, "top10": top10, "sep": sep},
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_fusion_validation_result.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
