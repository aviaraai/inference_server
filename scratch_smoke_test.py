"""
Real end-to-end smoke test: boots the FULL app (register/search/health,
morphology, duplicate-detection changes, all merged together) with real
GodhaarModel weights and a fresh empty FAISS index, then actually exercises
/register -> /search against real cattle photos.
"""
import os
import sys

os.environ.setdefault("MODEL_PATH", r"D:\Group Projects\Godhaar\Wildlife\for_aditya\best_top1.pt")
os.environ.setdefault("FAISS_INDEX_PATH", r"D:\Group Projects\inference_server\scratch_smoke_faiss.index")

from fastapi.testclient import TestClient
from main import app

FRONT1 = r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\2.png"
FRONT2 = r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\3.png"
MUZZLE2 = r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\5.png"
MUZZLE3 = r"C:\Users\Asus\.claude\image-cache\45070f9b-0eff-42ed-a955-a1a985a740d0\6.png"

ok = True

with TestClient(app) as client:
    print("=== /health ===")
    r = client.get("/health")
    print(r.status_code, r.json())
    if r.status_code != 200:
        ok = False

    print("\n=== /register ===")
    files = [
        ("front_images", ("front1.png", open(FRONT1, "rb"), "image/png")),
        ("front_images", ("front2.png", open(FRONT2, "rb"), "image/png")),
        ("muzzle_images", ("muzzle2.png", open(MUZZLE2, "rb"), "image/png")),
        ("muzzle_images", ("muzzle3.png", open(MUZZLE3, "rb"), "image/png")),
        # Reusing muzzle2 as the 3rd slot -- pragmatic for a smoke test
        # (proves the endpoint runs end-to-end), not an accuracy test.
        ("muzzle_images", ("muzzle2_dup.png", open(MUZZLE2, "rb"), "image/png")),
    ]
    data = {"candidates": "[]"}
    r = client.post("/register", files=files, data=data)
    print(r.status_code)
    import json as json_mod
    body = r.json()
    print(json_mod.dumps(body, indent=2))

    if r.status_code != 201:
        print("FAIL: register did not return 201")
        ok = False
    else:
        if "horn_shape" not in body:
            print("FAIL: horn_shape missing from RegisterResponse")
            ok = False
        if "video_url" in body or "output_video" in body:
            print("FAIL: unexpected cctv-shaped field leaked into RegisterResponse")
            ok = False
        embedding_ids = body["embedding_ids"]
        faiss_id = embedding_ids[0]
        new_body_color = body["extracted_colors"]["body"]["label"]
        new_muzzle_color = body["extracted_colors"]["muzzle"]["label"]

        print("\n=== /search (self-search against what we just registered) ===")
        files2 = [
            ("muzzle", ("muzzle2.png", open(MUZZLE2, "rb"), "image/png")),
            ("front", ("front1.png", open(FRONT1, "rb"), "image/png")),
        ]
        candidates = [{
            "faiss_id": faiss_id,
            "body_color": new_body_color,
            "muzzle_color": new_muzzle_color,
            "horn_shape": body.get("horn_shape"),
        }]
        data2 = {"top_k": "5", "candidates": json_mod.dumps(candidates)}
        r2 = client.post("/search", files=files2, data=data2)
        print(r2.status_code)
        body2 = r2.json()
        print(json_mod.dumps(body2, indent=2))

        if r2.status_code != 200:
            print("FAIL: search did not return 200")
            ok = False
        else:
            if "horn_shape" not in body2:
                print("FAIL: horn_shape missing from SearchResponse")
                ok = False
            top = body2["top_matches"][0] if body2["top_matches"] else None
            if not top or top["faiss_id"] != faiss_id:
                print(f"FAIL: self-search didn't return the just-registered faiss_id as a match: {top}")
                ok = False
            elif top["score"] < 0.9:
                print(f"WARN: self-search score lower than expected for the exact same photo: {top['score']}")
            else:
                print(f"PASS: self-search found faiss_id={faiss_id} at score={top['score']:.4f}")

        print("\n=== /register again (same animal -> should hit new DUPLICATE_ANIMAL error contract) ===")
        files3 = [
            ("front_images", ("front1.png", open(FRONT1, "rb"), "image/png")),
            ("front_images", ("front2.png", open(FRONT2, "rb"), "image/png")),
            ("muzzle_images", ("muzzle2.png", open(MUZZLE2, "rb"), "image/png")),
            ("muzzle_images", ("muzzle3.png", open(MUZZLE3, "rb"), "image/png")),
            ("muzzle_images", ("muzzle2_dup.png", open(MUZZLE2, "rb"), "image/png")),
        ]
        data3 = {"candidates": json_mod.dumps(candidates)}
        r3 = client.post("/register", files=files3, data=data3)
        print(r3.status_code)
        body3 = r3.json()
        print(json_mod.dumps(body3, indent=2))

        if r3.status_code != 409:
            print(f"FAIL: expected 409 duplicate, got {r3.status_code}")
            ok = False
        elif body3.get("error_code") != "DUPLICATE_ANIMAL":
            print(f"FAIL: expected error_code=DUPLICATE_ANIMAL, got {body3.get('error_code')}")
            ok = False
        elif "detail" not in body3 or body3["detail"].get("matched_faiss_id") != faiss_id:
            print(f"FAIL: detail.matched_faiss_id missing or wrong: {body3.get('detail')}")
            ok = False
        else:
            print(f"PASS: new error contract confirmed -> error_code=DUPLICATE_ANIMAL, "
                  f"detail.matched_faiss_id={body3['detail']['matched_faiss_id']}")

print("\n=== OVERALL:", "PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
