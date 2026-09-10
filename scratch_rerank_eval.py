"""scratch_rerank_eval.py — STAGE 1 GATE for promoting LightGlue from a
demote-only veto to a top-K re-ranker.

Question: the fused+whitened embedding puts the correct animal in the top-10
95.5% of the time (scratch_ablation_fusion.py) but only gets top-1 right
77-89% of the time — and LightGlue separates same-from-different at AUC 0.961
(scratch_lightglue_frequency_probe.py). Does actually RE-RANKING the top-K
embedding candidates by LightGlue evidence close some of that gap, measured
end-to-end, or does it not move the needle enough to justify the added
latency and complexity?

Protocol (222-animal Uttarakhand population, godhaar/ only -- same
population every prior scratch script in this repo uses, ~1% duplication,
independently confirmed clean by the Stage-6/Stage-4 investigations):
  1. Embed every UKDE*_muzzle*.jpg that passes quality_check + crop_cattle
     with the REAL production fusion encoder (full photo, no crop -- Stage 3
     of the fusion upgrade) + frozen whitening.
  2. Also crop_cattle() + DISK-extract each image ONCE -- this simulates
     what MUZZLE_CROP_CACHE_DIR would hold for that photo if it were a
     registered candidate (a crop + its DISK features), and what a query
     photo's own crop would be at search time.
  3. Leave-one-out: for each query image, rank every OTHER animal by max
     cosine embedding score (matches production's cattleScores[gid] =
     max(...) aggregation in go-apiserver/routes.go) over its own OTHER
     photos. For each of the top-K candidate animals, LightGlue-match the
     query's crop against the crop belonging to whichever OTHER photo gave
     that animal its max score (the photo that WOULD be the registered
     embedding backing that candidate's rank).
  4. Re-rank the top-K candidates by combining embedding rank + LightGlue
     evidence rank (see Stage 2 in the task -- reciprocal rank fusion, not a
     raw-score sum; the exact combination lives in rrf_combine() below,
     shared with production code once this gate passes).
  5. Report baseline top-1, re-ranked top-1 at each K, and the oracle
     (top-K recall = ceiling any re-ranker could reach), plus measured
     latency per search at each K.

This is a MEASUREMENT gate. If (b) does not clearly beat (a), stop --
do not build the production path in main.py/decision.go.
"""
import collections
import glob
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
from pipeline import lightglue_verify as lg

DATA = r"D:\Group Projects\godhaar-all-images\godhaar"
DINO_MODEL_PATH = r"D:\Group Projects\inference_server\appstorage\Models\model.pt"
RESNET_PATH = r"D:\Group Projects\inference_server\appstorage\Models\resnet50\resnet50_imagenet1k_v2.pth"
WHITENING_PATH = r"D:\Group Projects\inference_server\appstorage\Models\whitening\whitening_v1.npz"
YOLO_PATH = r"D:\Group Projects\inference_server\appstorage\Models\yolov8s.pt"
KS = (3, 5, 10, 20)
RRF_C = 60  # standard reciprocal-rank-fusion constant


def tag_of(p):
    return os.path.basename(p).split("_muzzle")[0]


def rrf_combine(embedding_rank: int, lightglue_rank: int | None, c: int = RRF_C) -> float:
    """Reciprocal rank fusion. embedding_rank/lightglue_rank are 1-indexed
    positions WITHIN the top-K candidate set being re-ranked (not the full
    gallery). A missing LightGlue rank (no cached crop for that candidate --
    see Stage 2's graceful-degradation requirement) contributes 0 rather
    than the worst possible rank, so a candidate with no cached crop keeps
    its embedding-derived position relative to OTHER crop-less candidates
    instead of being punished for a missing cache entry it had no control
    over. This is shared verbatim with the production implementation once
    Stage 1 passes -- do not fork logic between the eval and the real path.
    """
    score = 1.0 / (c + embedding_rank)
    if lightglue_rank is not None:
        score += 1.0 / (c + lightglue_rank)
    return score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    dino, ck = GodhaarModel.load_checkpoint(DINO_MODEL_PATH, device=device)
    dino.eval()
    resnet = load_resnet50(RESNET_PATH, device)
    whitening = WhiteningTransform.load(WHITENING_PATH)
    encoder = FusionEncoder(dino, resnet, whitening, device)
    load_yolo(YOLO_PATH)
    lg.load_lightglue()
    print(f"encoders + LightGlue ready (dino epoch={ck.get('epoch')}, lightglue available={lg.available()})", flush=True)

    all_paths = sorted(glob.glob(os.path.join(DATA, "UKDE*_muzzle*.jpg")))
    print(f"{len(all_paths)} muzzle photos found", flush=True)

    paths, embs, crops_feats, tags = [], [], [], []
    t0 = time.time()
    for n, p in enumerate(all_paths):
        raw = open(p, "rb").read()
        if quality_check(raw)[0] not in ("OK", "GOOD"):
            continue
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        crop, det, _conf = crop_cattle(bgr)
        if crop is None:
            continue
        e = embed_batch([raw], encoder, device).numpy()[0]
        feats = lg.extract_features(crop)  # DISK features on the crop, kept on-device
        paths.append(p)
        embs.append(e)
        crops_feats.append(feats)
        tags.append(tag_of(p))
        if (n + 1) % 150 == 0:
            print(f"  prepared {n+1}/{len(all_paths)} ({time.time()-t0:.0f}s)", flush=True)

    E = np.array(embs, dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    tags = np.array(tags)
    n_img = len(paths)
    print(f"prepared {n_img} images / {len(set(tags.tolist()))} animals ({time.time()-t0:.0f}s)\n", flush=True)

    cnt = collections.Counter(tags.tolist())
    usable = {t for t, c in cnt.items() if c >= 2}  # need >=1 gallery photo after LOO

    S = E @ E.T

    baseline_correct = 0
    oracle_correct = {K: 0 for K in KS}
    rerank_correct = {K: 0 for K in KS}
    latency_ms = {K: [] for K in KS}
    n_queries = 0

    max_k = max(KS)
    for i in range(n_img):
        if tags[i] not in usable:
            continue
        n_queries += 1

        # Per-animal max embedding score over this query's OTHER photos (mirrors
        # go-apiserver's cattleScores[gid] = max(...) aggregation), tracking
        # WHICH photo j produced each animal's max (that's the "registered
        # embedding" this candidate would be pinned to).
        per_animal_best = {}  # tag -> (score, j)
        for j in range(n_img):
            if j == i:
                continue
            s = float(S[i, j])
            cur = per_animal_best.get(tags[j])
            if cur is None or s > cur[0]:
                per_animal_best[tags[j]] = (s, j)

        ranked = sorted(per_animal_best.items(), key=lambda kv: -kv[1][0])  # [(tag, (score, j)), ...]
        ranked_tags = [t for t, _ in ranked]

        if ranked_tags and ranked_tags[0] == tags[i]:
            baseline_correct += 1

        for K in KS:
            topk = ranked[:K]
            if tags[i] in [t for t, _ in topk]:
                oracle_correct[K] += 1

            t_start = time.perf_counter()
            lg_ranks = []  # (tag, match_ratio, num_matches)
            for rank_idx, (t, (score, j)) in enumerate(topk, start=1):
                try:
                    result = lg._match_feature_sets(crops_feats[i], crops_feats[j])
                    lg_ranks.append((t, result["match_ratio"], result["num_matches"]))
                except Exception:
                    lg_ranks.append((t, None, None))
            elapsed = (time.perf_counter() - t_start) * 1000
            latency_ms[K].append(elapsed)

            # LightGlue rank within this K, by match_ratio descending (ties by num_matches).
            with_lg = [(t, r, m) for t, r, m in lg_ranks if r is not None]
            with_lg.sort(key=lambda x: (-x[1], -x[2]))
            lg_rank_of = {t: idx + 1 for idx, (t, r, m) in enumerate(with_lg)}

            combo = []
            for rank_idx, (t, (score, j)) in enumerate(topk, start=1):
                combo.append((t, rrf_combine(rank_idx, lg_rank_of.get(t))))
            combo.sort(key=lambda kv: -kv[1])
            new_top1 = combo[0][0]
            if new_top1 == tags[i]:
                rerank_correct[K] += 1

        if n_queries % 100 == 0:
            print(f"  {n_queries} queries processed ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print(f"STAGE 1 RESULTS — {n_queries} leave-one-out queries, {len(usable)} animals")
    print("=" * 78)
    print(f"(a) BASELINE top-1 (fused+whitened embedding alone): "
          f"{baseline_correct}/{n_queries} = {100*baseline_correct/n_queries:.1f}%")
    print()
    print(f"{'K':>4} {'(b) re-ranked top-1':>22} {'(c) oracle (top-K recall)':>28} {'gap (b) vs (a)':>16} "
          f"{'median latency/search (ms)':>28}")
    for K in KS:
        b = 100 * rerank_correct[K] / n_queries
        c = 100 * oracle_correct[K] / n_queries
        lat = np.median(latency_ms[K])
        print(f"{K:>4} {b:>21.1f}% {c:>27.1f}% {b-100*baseline_correct/n_queries:>+15.1f}% {lat:>27.1f}")

    result = {
        "n_queries": n_queries,
        "n_animals": len(usable),
        "baseline_top1_pct": 100 * baseline_correct / n_queries,
        "by_k": {
            str(K): {
                "rerank_top1_pct": 100 * rerank_correct[K] / n_queries,
                "oracle_topk_recall_pct": 100 * oracle_correct[K] / n_queries,
                "median_latency_ms": float(np.median(latency_ms[K])),
                "p95_latency_ms": float(np.percentile(latency_ms[K], 95)),
            }
            for K in KS
        },
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_rerank_eval_result.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
