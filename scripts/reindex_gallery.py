"""
scripts/reindex_gallery.py — STAGE 5: re-embed the registered gallery with
the new fusion+PCA-whitening encoder and rebuild the FAISS index.

Gallery and query MUST share preprocessing. Once main.py's lifespan loads
the fusion encoder (Stage 2/3), every live /search call embeds its query
photo with DINOv2+ResNet50x2+whitening -- an index still full of plain-
DINOv2 vectors is comparing apples to oranges on every single request. This
script is what closes that gap: a full, resumable, dry-run-by-default
migration of one FAISS index to the new embedding space, preserving each
embedding's original faiss_id (go-apiserver's DB references it directly --
see add_batch_with_ids's docstring in faiss_index.py for why minting new
ids would silently break every stored mapping).

Image source
------------
inference_server has no DB connection and no object-storage credentials
(same limitation documented in scripts/backfill_muzzle_cache.py) -- it
cannot enumerate registered animals or fetch their original photos itself.
Two source modes are supported:

  --api-url/--token   Same go-apiserver debug endpoint backfill_muzzle_cache
                       .py already uses (GET {api-url}/debug/muzzle-embeddings
                       -> [{faiss_id, godhaar_id, sequence, image_url}, ...]).
                       This is the REAL production path.

  --local-manifest     A JSON file of the same row shape but with
                       "image_path" (local file) instead of "image_url" --
                       for testing/dry-runs against a local image set (e.g.
                       this repo's own godhaar-all-images corpus stood in
                       as a fake "gallery") without a live API/token.
                       NOT how a real migration should be sourced.

CRITICAL: the image fetched here must be the FULL photo, not a crop --
crop_cattle() output would put the gallery in a different distribution than
Stage 3's query-side embedding (which also stopped cropping). Do not
"helpfully" crop this script's input to match old behavior.

Resume support
--------------
Progress is checkpointed to `<out-index>.progress.json` (a list of faiss_ids
already embedded+staged). A crash or Ctrl-C partway through can be resumed
by re-running the identical command -- already-done faiss_ids are skipped
and their previously-computed vectors (persisted in the SAME checkpoint
file) are reused rather than re-downloaded/re-embedded.

Dry-run
-------
Default. Reports what would be re-embedded (counts, distinct animals) and
does nothing else. --execute actually downloads, embeds, and writes a NEW
index file (never overwrites FAISS_INDEX_PATH directly) -- operators must
explicitly point FAISS_INDEX_PATH at the new file once satisfied, the same
"don't auto-swap the live thing" posture as every other risky script in
this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faiss_index import FaissIndex  # noqa: E402
from godhaar.config import EMB_DIM, MODEL_VERSION, RESNET_MODEL_PATH, WHITENING_MODEL_PATH  # noqa: E402
from godhaar.model import GodhaarModel  # noqa: E402
from pipeline.fusion_encoder import FusionEncoder, load_resnet50  # noqa: E402
from pipeline.muzzle import embed_batch  # noqa: E402
from pipeline.whitening import WhiteningTransform  # noqa: E402


def _api_get(url: str, token: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_bytes(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    download failed: {e}")
        return None


def _load_rows(args) -> list[dict]:
    if args.local_manifest:
        rows = json.loads(open(args.local_manifest, "r", encoding="utf-8").read())
        for r in rows:
            r.setdefault("image_url", None)
        return rows
    if not args.api_url or not args.token:
        raise SystemExit("--api-url/--token (or GODHAAR_API_URL/GODHAAR_API_TOKEN), or --local-manifest, is required")
    print(f"Fetching registered muzzle embeddings from {args.api_url}/debug/muzzle-embeddings ...")
    return _api_get(f"{args.api_url}/debug/muzzle-embeddings", args.token)


def _read_image_bytes(row: dict) -> bytes | None:
    if row.get("image_path"):
        try:
            return open(row["image_path"], "rb").read()
        except OSError as e:
            print(f"    local read failed: {e}")
            return None
    if row.get("image_url"):
        return _download_bytes(row["image_url"])
    return None


def _load_progress(path: str) -> dict:
    if os.path.exists(path):
        return json.loads(open(path, "r", encoding="utf-8").read())
    return {"done": {}}  # faiss_id (str) -> embedding (list[float])


def _save_progress(path: str, progress: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", default=os.environ.get("GODHAAR_API_URL"))
    p.add_argument("--token", default=os.environ.get("GODHAAR_API_TOKEN"))
    p.add_argument("--local-manifest", default=None,
                    help="JSON file of {faiss_id, godhaar_id, sequence, image_path} rows -- test/dry-run only")
    p.add_argument("--dino-model-path", default=os.environ.get("MODEL_PATH", "/appstorage/Models/model.pt"))
    p.add_argument("--resnet-model-path", default=os.environ.get("RESNET_MODEL_PATH", RESNET_MODEL_PATH))
    p.add_argument("--whitening-path", default=os.environ.get("WHITENING_MODEL_PATH", WHITENING_MODEL_PATH))
    p.add_argument("--out-index", required=True, help="Path to write the NEW faiss index (never overwrites the live one)")
    p.add_argument("--execute", action="store_true", help="Actually download+embed+write. Default is report-only.")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    rows = _load_rows(args)
    print(f"{len(rows)} registered muzzle embeddings on record.\n")

    by_animal: dict[str, list[dict]] = {}
    for r in rows:
        by_animal.setdefault(r["godhaar_id"], []).append(r)

    print("=" * 78)
    print("REPORT")
    print("=" * 78)
    print(f"  distinct faiss_ids (embeddings) to re-embed: {len(rows)}")
    print(f"  distinct animals:                            {len(by_animal)}")

    if not args.execute:
        print("\nDry run only -- nothing downloaded, embedded, or written. Re-run with --execute to apply.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out_index)), exist_ok=True)
    progress_path = args.out_index + ".progress.json"
    progress = _load_progress(progress_path)
    done_ids = set(progress["done"].keys())
    remaining = [r for r in rows if str(r["faiss_id"]) not in done_ids]
    print(f"\nResume state: {len(done_ids)} already embedded (checkpoint={progress_path}), "
          f"{len(remaining)} remaining.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino, ck = GodhaarModel.load_checkpoint(args.dino_model_path, device=device)
    dino.eval()
    resnet = load_resnet50(args.resnet_model_path, device)
    whitening = WhiteningTransform.load(args.whitening_path)
    encoder = FusionEncoder(dino, resnet, whitening, device)
    encoder.eval()
    print(f"Fusion encoder ready (dino epoch={ck.get('epoch')}, "
          f"whitening hash={whitening.content_hash[:12]}...)\n")

    t0 = time.time()
    n_ok = n_fail = 0
    for i, row in enumerate(remaining):
        fid = row["faiss_id"]
        img_bytes = _read_image_bytes(row)
        if img_bytes is None:
            print(f"  faiss_id={fid}: SKIP (could not fetch image)")
            n_fail += 1
            continue
        try:
            emb = embed_batch([img_bytes], encoder, device).numpy()[0]  # (EMB_DIM,), full photo, no crop
        except Exception as e:
            print(f"  faiss_id={fid}: SKIP (embed failed: {e})")
            n_fail += 1
            continue
        assert emb.shape == (EMB_DIM,), f"expected ({EMB_DIM},), got {emb.shape} -- EMB_DIM contract broken"
        progress["done"][str(fid)] = emb.tolist()
        n_ok += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(remaining):
            _save_progress(progress_path, progress)
            print(f"  {i+1}/{len(remaining)} embedded ({n_ok} ok, {n_fail} failed) — {time.time()-t0:.0f}s", flush=True)

    _save_progress(progress_path, progress)
    print(f"\nEmbedding pass done: {n_ok} ok, {n_fail} failed/skipped this run "
          f"({len(progress['done'])} total in checkpoint).")

    if n_fail:
        print(
            f"\n{n_fail} embeddings failed this run -- re-run the identical command to retry "
            f"just those (everything in the checkpoint is skipped). Do NOT build the index "
            f"below until failures are resolved or explicitly accepted as gaps."
        )

    print(f"\nBuilding new FAISS index at {args.out_index} ...")
    new_index = FaissIndex(embedding_dim=EMB_DIM)
    all_ids = [int(fid) for fid in progress["done"].keys()]
    all_vecs = np.array([progress["done"][str(fid)] for fid in all_ids], dtype=np.float32)
    if len(all_ids) == 0:
        print("Nothing embedded -- refusing to write an empty index.")
        sys.exit(1)

    # add_batch_with_ids in chunks to keep memory bounded on very large galleries.
    B = args.batch_size
    import asyncio

    async def _build():
        for start in range(0, len(all_ids), B):
            await new_index.add_batch_with_ids(
                all_vecs[start:start + B], all_ids[start:start + B]
            )

    asyncio.run(_build())
    new_index._model_version = MODEL_VERSION  # so save() writes the NEW version into meta.json

    async def _save():
        await new_index.save(args.out_index)

    asyncio.run(_save())

    print(f"\nDone. {len(new_index)} vectors written to {args.out_index} "
          f"(meta.json records model_version={MODEL_VERSION}).")
    print(
        "This does NOT touch the live FAISS_INDEX_PATH. Once satisfied this is "
        "correct, point FAISS_INDEX_PATH at this new file and restart the server "
        "-- faiss_index.py's load() will confirm the model_version now matches."
    )


if __name__ == "__main__":
    main()
