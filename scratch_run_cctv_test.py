"""
One-off test script: run the 48.8s clip through the CCTV pipeline via the
real HTTP endpoints (mounting only cctv_router, so it doesn't need
GodhaarModel/FAISS which the full app's lifespan requires but CCTV never
touches) and check the fix-verification success criteria.
"""
import json
import subprocess
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cctv.routes import router as cctv_router
from cctv.config import DB_PATH
import cctv.database as db

VIDEO_PATH = r"C:\Users\Asus\Downloads\document_6316777104247628263.mp4"

app = FastAPI()
app.include_router(cctv_router)
client = TestClient(app)

print(f"Using video: {VIDEO_PATH}")

with open(VIDEO_PATH, "rb") as f:
    resp = client.post(
        "/cctv/analyze",
        files={"video": ("test_clip.mp4", f, "video/mp4")},
        data={"preset": "crowded", "enable_analytics": "true"},
    )
resp.raise_for_status()
job_id = resp.json()["job_id"]
print(f"job_id = {job_id}")

t0 = time.time()
while True:
    s = client.get(f"/cctv/jobs/{job_id}/status").json()
    print(f"  status={s.get('status')} progress={s.get('progress')} "
          f"frames={s.get('frames_processed')}/{s.get('total_frames')} "
          f"elapsed={time.time()-t0:.1f}s")
    if s.get("status") in ("done", "failed"):
        break
    if time.time() - t0 > 900:
        print("TIMEOUT waiting for job to finish")
        sys.exit(1)
    time.sleep(3)

if s.get("status") == "failed":
    print("JOB FAILED:", s.get("error"))
    sys.exit(1)

result = client.get(f"/cctv/jobs/{job_id}/result").json()
analytics = client.get(f"/cctv/jobs/{job_id}/analytics").json()
history = client.get("/cctv/history").json()
trends = client.get("/cctv/trends").json()

print("\n=== /result ===")
print(json.dumps(result, indent=2)[:2000])
print("\n=== /analytics (trimmed) ===")
print(json.dumps({k: v for k, v in analytics.items() if k != "per_cow"}, indent=2))
print("\n=== /history (matching job) ===")
hist_row = next((h for h in history if h["job_id"] == job_id), None)
print(json.dumps(hist_row, indent=2))

# ── checks ──────────────────────────────────────────────────────────
print("\n=== CHECKS ===")
final_count = result["final_cattle_count"]
analytics_count = analytics["total_cattle"]
history_count = hist_row["final_cattle_count"] if hist_row else None
tracked_result = result["unique_tracked_cattle"]
tracked_analytics = analytics["unique_tracked_cattle"]
tracked_history = hist_row["unique_tracked_cattle"] if hist_row else None
trend_row = next((t for t in trends if t["job_id"] == job_id), None)

ok = True

print(f"[INFO] peak={final_count}  tracked={tracked_result}")

if final_count == analytics_count:
    print(f"[PASS] /result peak ({final_count}) == /analytics peak ({analytics_count})")
else:
    print(f"[FAIL] /result peak ({final_count}) != /analytics peak ({analytics_count})")
    ok = False

if history_count == final_count:
    print(f"[PASS] /history peak ({history_count}) == /result peak ({final_count})")
else:
    print(f"[FAIL] /history peak ({history_count}) != /result peak ({final_count})")
    ok = False

if tracked_result == tracked_analytics:
    print(f"[PASS] /result tracked ({tracked_result}) == /analytics tracked ({tracked_analytics})")
else:
    print(f"[FAIL] /result tracked ({tracked_result}) != /analytics tracked ({tracked_analytics})")
    ok = False

if tracked_history == tracked_result:
    print(f"[PASS] /history tracked ({tracked_history}) == /result tracked ({tracked_result})")
else:
    print(f"[FAIL] /history tracked ({tracked_history}) != /result tracked ({tracked_result})")
    ok = False

if trend_row and trend_row.get("count") == final_count and trend_row.get("unique_tracked_cattle") == tracked_result:
    print(f"[PASS] /trends peak+tracked match ({trend_row.get('count')}, {trend_row.get('unique_tracked_cattle')})")
else:
    print(f"[FAIL] /trends row mismatch or missing: {trend_row}")
    ok = False

import os
import tempfile
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()


def _fetch_and_probe(url: str, label: str) -> bool:
    r = client.get(url)
    if r.status_code != 200 or len(r.content) == 0:
        print(f"[FAIL] {label}: GET {url} -> {r.status_code}, {len(r.content)} bytes")
        return False
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.write(fd, r.content)
    os.close(fd)
    probe = subprocess.run([ffmpeg_path, "-i", tmp_path], capture_output=True, text=True)
    codec_line = next((l for l in probe.stderr.splitlines() if "Video:" in l), "")
    os.unlink(tmp_path)
    if "h264" in codec_line.lower():
        print(f"[PASS] {label}: GET {url} -> {len(r.content)} bytes, h264 confirmed")
        return True
    print(f"[FAIL] {label}: GET {url} -> not h264 ({codec_line.strip()})")
    return False


print(f"\n[INFO] result.video_url = {result.get('video_url')}")
print(f"[INFO] history.video_url = {hist_row.get('video_url') if hist_row else None}")

if not _fetch_and_probe(result.get("video_url", ""), "video via /result"):
    ok = False

# Restart-resilience: clear the in-memory job tracker (simulates a server
# restart) and confirm the SAME video_url still serves the file, falling
# back to the sessions table + on-disk file.
import cctv.routes as routes_mod
with routes_mod._lock:
    routes_mod._jobs.clear()

if not _fetch_and_probe(hist_row.get("video_url", ""), "video via /history after simulated restart"):
    ok = False

print("\n=== OVERALL:", "PASS" if ok else "FAIL", "===")
sys.exit(0 if ok else 1)
